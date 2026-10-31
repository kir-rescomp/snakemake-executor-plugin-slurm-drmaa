"""End-to-end tests: run Snakemake with this executor and a stand-in drmaa."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

FAKE_DRMAA = Path(__file__).parent / "fake_drmaa"

SNAKEFILE = """
rule all:
    input: "cpu.txt", "gpu.txt"

rule cpu:
    output: "cpu.txt"
    threads: 2
    resources:
        slurm_account="kir.prj", slurm_partition="short", runtime=30, mem_mb=1000
    shell: "echo cpu > {output}"

rule gpu:
    output: "gpu.txt"
    resources:
        slurm_partition="gpu_interactive", gpu=1, runtime=5, account="gpu_kir.prj"
    shell: "echo gpu > {output}"
"""


def run_snakemake(workdir: Path, env_extra: dict, *args):
    (workdir / "Snakefile").write_text(SNAKEFILE)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(FAKE_DRMAA), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["FAKE_DRMAA_RECORD"] = str(workdir / "specs.txt")
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "snakemake", "--executor", "slurm-drmaa",
         "--jobs", "2", "--slurm-drmaa-log-dir", "logs", *args],
        cwd=workdir, env=env, capture_output=True, text=True, timeout=300,
    )


def recorded_specs(workdir: Path):
    return sorted((workdir / "specs.txt").read_text().splitlines())


def test_jobs_run_with_specs_built_from_resources(tmp_path):
    result = run_snakemake(tmp_path, {})
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cpu.txt").read_text() == "cpu\n"
    assert (tmp_path / "gpu.txt").read_text() == "gpu\n"
    specs = recorded_specs(tmp_path)
    assert (
        "--account=kir.prj --partition=short --time=00:30:00 "
        "--mem=1000 --cpus-per-task=2"
    ) in specs
    gpu_spec = next(s for s in specs if "gres" in s)
    assert "--partition=gpu_interactive" in gpu_spec
    assert "--time=00:05:00" in gpu_spec
    assert gpu_spec.endswith("--gres=gres/gpu:1")
    # The unread legacy 'account' resource is reported, not silently ignored.
    assert "Resource 'account' is not read" in result.stderr


def test_rejected_submission_reports_spec_and_slurm_reason(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        '#!/bin/sh\n[ "$1" = --version ] && { echo "slurm 23.11.10"; exit 0; }\n'
        "echo 'sbatch: error: QOSMinGRES' >&2\nexit 1\n"
    )
    sbatch.chmod(0o755)
    result = run_snakemake(
        tmp_path,
        {"FAKE_DRMAA_REJECT": "gres/gpu", "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        "--keep-going",
    )
    assert result.returncode != 0
    assert "Native specification: --partition=gpu_interactive" in result.stderr
    assert "QOSMinGRES" in result.stderr
    # The CPU job is unaffected by the GPU job's rejection.
    assert (tmp_path / "cpu.txt").exists()


def test_invalid_resources_fail_before_submission(tmp_path):
    (tmp_path / "Snakefile").write_text(
        'rule bad:\n    output: "x"\n    resources: gpu=1, gres="gpu:1"\n'
        '    shell: "touch {output}"\n'
    )
    env = dict(os.environ, PYTHONPATH=str(FAKE_DRMAA),
               FAKE_DRMAA_RECORD=str(tmp_path / "specs.txt"))
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "--executor", "slurm-drmaa", "--jobs", "1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode != 0
    assert "not both" in result.stderr
    assert not (tmp_path / "specs.txt").exists()


def test_non_slurm_drmaa_library_is_refused(tmp_path):
    result = run_snakemake(tmp_path, {"FAKE_DRMAA_IMPL": "PBS Pro DRMAA 1.0"})
    assert result.returncode != 0
    assert "requires the Slurm DRMAA library" in result.stderr


def fake_sbatch_version(tmp_path: Path, version: str) -> dict:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(f'#!/bin/sh\necho "{version}"\n')
    sbatch.chmod(0o755)
    return {"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}


def test_gres_prefix_detected_from_older_slurm(tmp_path):
    result = run_snakemake(tmp_path, fake_sbatch_version(tmp_path, "slurm 22.05.9"))
    assert result.returncode == 0, result.stderr
    assert any(s.endswith("--gres=gres:gpu:1") for s in recorded_specs(tmp_path))


def test_gres_prefix_setting_overrides_detection(tmp_path):
    result = run_snakemake(
        tmp_path,
        fake_sbatch_version(tmp_path, "slurm 23.11.10"),
        "--slurm-drmaa-gres-prefix",
        "none",
    )
    assert result.returncode == 0, result.stderr
    assert any(s.endswith("--gres=gpu:1") for s in recorded_specs(tmp_path))


def test_undetectable_release_warns_and_uses_default(tmp_path):
    result = run_snakemake(tmp_path, {})
    assert result.returncode == 0, result.stderr
    assert "Could not determine the Slurm release" in result.stderr
    assert any(s.endswith("--gres=gres/gpu:1") for s in recorded_specs(tmp_path))


def test_job_logs_are_files_and_failures_point_to_them(tmp_path):
    (tmp_path / "Snakefile").write_text(
        'rule fails:\n    output: "x"\n'
        '    shell: "echo marker-from-job; exit 3"\n'
    )
    env = dict(os.environ, PYTHONPATH=str(FAKE_DRMAA),
               FAKE_DRMAA_RECORD=str(tmp_path / "specs.txt"))
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "--executor", "slurm-drmaa", "--jobs", "1",
         "--slurm-drmaa-log-dir", "logs"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode != 0
    logs = list((tmp_path / "logs" / "fails").glob("*.log"))
    assert len(logs) == 1, result.stderr
    assert "marker-from-job" in logs[0].read_text()
    jobid = logs[0].stem
    assert f"Slurm job {jobid} failed." in result.stderr
    assert f"Job log: {logs[0]}" in result.stderr
    assert f"sacct -j {jobid}" in result.stderr


def test_default_log_dir(tmp_path):
    result = run_snakemake_without_log_dir(tmp_path)
    assert result.returncode == 0, result.stderr
    assert list((tmp_path / ".snakemake" / "slurm_drmaa_logs" / "cpu").glob("*.log"))


def run_snakemake_without_log_dir(workdir: Path):
    (workdir / "Snakefile").write_text(SNAKEFILE)
    env = dict(os.environ, PYTHONPATH=str(FAKE_DRMAA),
               FAKE_DRMAA_RECORD=str(workdir / "specs.txt"))
    return subprocess.run(
        [sys.executable, "-m", "snakemake", "--executor", "slurm-drmaa", "--jobs", "2"],
        cwd=workdir, env=env, capture_output=True, text=True, timeout=300,
    )
