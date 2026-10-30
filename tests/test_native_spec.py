import pytest
from snakemake_interface_common.exceptions import WorkflowError

from snakemake_executor_plugin_slurm_drmaa.native_spec import (
    build_native_spec,
    format_runtime,
    legacy_resource_warnings,
)


def spec(threads=1, extra=None, **resources):
    return build_native_spec(resources, threads, extra=extra)


def test_cpu_job():
    assert spec(
        threads=4,
        slurm_account="kir.prj",
        slurm_partition="short",
        runtime=240,
        mem_mb=16000,
    ) == (
        "--account=kir.prj --partition=short --time=04:00:00 "
        "--mem=16000 --cpus-per-task=4"
    )


def test_unset_resources_are_omitted():
    assert spec(threads=None) == ""
    assert spec(threads=2, qos=None, constraint="") == "--cpus-per-task=2"


@pytest.mark.parametrize(
    "minutes, expected",
    [(5, "00:05:00"), (90, "01:30:00"), (2880, "48:00:00"), (86400, "1440:00:00")],
)
def test_runtime_is_total_hours(minutes, expected):
    # slurm-drmaa discards days, so runtime must never use the D-HH form.
    assert format_runtime(minutes) == expected


@pytest.mark.parametrize("bad", [0, -1, "four hours"])
def test_invalid_runtime(bad):
    with pytest.raises(WorkflowError):
        format_runtime(bad)


def test_gres_gets_prefix():
    assert spec(gres="gpu:1") == "--cpus-per-task=1 --gres=gres/gpu:1"


def test_gres_prefix_not_doubled_and_multiple_entries():
    assert spec(threads=None, gres="gres/gpu:a100:2, shard:1") == (
        "--gres=gres/gpu:a100:2,gres/shard:1"
    )


def test_gpu_and_model_resources():
    assert spec(threads=None, gpu=2) == "--gres=gres/gpu:2"
    assert spec(threads=None, gpu=1, gpu_model="l4") == "--gres=gres/gpu:l4:1"


def test_gpu_combined_with_non_gpu_gres():
    assert spec(threads=None, gpu=1, gres="shard:1") == "--gres=gres/shard:1,gres/gpu:1"


@pytest.mark.parametrize(
    "resources",
    [
        {"gpu": 1, "gres": "gpu:1"},
        {"gpu_model": "l4"},
        {"gpu": 0},
        {"gres": "gpu 1"},
        {"cpus_per_gpu": 4},
        {"tasks_per_gpu": 1},
    ],
)
def test_invalid_gpu_requests(resources):
    with pytest.raises(WorkflowError):
        spec(**resources)


def test_mem_per_cpu_takes_precedence():
    assert spec(threads=None, mem_mb=8000, mem_mb_per_cpu=2000) == "--mem-per-cpu=2000"


def test_cpus_per_task_overrides_threads():
    assert spec(threads=8, cpus_per_task=2) == "--cpus-per-task=2"


def test_task_layout_and_scheduling_options():
    assert spec(
        threads=None, nodes=1, tasks=4, tasks_per_node=4, constraint="avx2", qos="high"
    ) == "--nodes=1 --ntasks=4 --ntasks-per-node=4 --constraint=avx2 --qos=high"


@pytest.mark.parametrize("value", ["gpu short", "'short'", 'a"b'])
def test_values_with_whitespace_or_quotes_are_rejected(value):
    with pytest.raises(WorkflowError):
        spec(slurm_partition=value)


def test_extra_arguments_are_appended():
    assert spec(
        threads=None, slurm_partition="short", slurm_extra="--comment=x", extra="--requeue"
    ) == "--partition=short --comment=x --requeue"


@pytest.mark.parametrize(
    "extra",
    [
        "--gpus=1",
        "--gpus-per-node=1",
        "-G 1",
        "--gres=gpu:1",
        "--comment=x --cpus-per-gpu=2",
        "--comment='quoted'",
    ],
)
def test_gpu_options_and_quotes_rejected_in_extra(extra):
    with pytest.raises(WorkflowError):
        spec(extra=extra)
    with pytest.raises(WorkflowError):
        spec(slurm_extra=extra)


def test_legacy_resource_warnings():
    warnings = legacy_resource_warnings(
        {"account": "kir.prj", "partition": "short", "time": "04:00:00"}
    )
    assert len(warnings) == 3
    assert "slurm_account" in warnings[0]
    # No warning once the current name is also set.
    assert legacy_resource_warnings({"partition": "short", "slurm_partition": "short"}) == []


@pytest.mark.parametrize(
    "version, prefix",
    [
        ("slurm 20.11.9", ""),
        ("slurm 21.08.0", "gres:"),
        ("slurm 23.02.7", "gres:"),
        ("slurm 23.11.10", "gres/"),
        ("slurm 25.05.3", "gres/"),
        ("not a version", None),
        (None, None),
    ],
)
def test_gres_prefix_follows_slurm_release(version, prefix):
    from snakemake_executor_plugin_slurm_drmaa.native_spec import (
        gres_prefix_for_slurm_version,
    )

    assert gres_prefix_for_slurm_version(version) == prefix


@pytest.mark.parametrize(
    "prefix, expected",
    [("gres/", "--gres=gres/gpu:1"), ("gres:", "--gres=gres:gpu:1"), ("", "--gres=gpu:1")],
)
def test_gres_prefix_is_applied_and_existing_prefixes_replaced(prefix, expected):
    for gres in ("gpu:1", "gres/gpu:1", "gres:gpu:1"):
        assert build_native_spec({"gres": gres}, None, gres_prefix=prefix) == expected
