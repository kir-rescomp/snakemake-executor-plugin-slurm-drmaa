"""Build a slurm-drmaa native specification from Snakemake job resources.

slurm-drmaa, the C library that implements DRMAA on Slurm clusters, does not
call ``sbatch``. It parses its own subset of ``sbatch`` options and builds the
job request itself. Three of its behaviours shape this module; each was
confirmed in the slurm-drmaa and drmaa-utils sources:

* The native specification is split on spaces and tabs only, with no quote
  handling. Option values therefore must not contain whitespace or quotes.
* ``--gres`` is the only GPU option it accepts; ``--gpus``, ``--gpus-per-node``
  and related options are rejected as unsupported. The ``--gres`` value is
  copied verbatim into the job request, whereas ``sbatch`` first adds a type
  prefix to each entry (``gres_prepend_tres_type`` in Slurm). Without the
  prefix the controller does not recognise the request, and the job is
  rejected. The prefix depends on the Slurm release: none up to 20.11,
  ``gres:`` from 21.08 to 23.02, and ``gres/`` from 23.11 (see
  ``gres_prefix_for_slurm_version``). This module adds the prefix itself.
* The time limit must be written as ``H:MM`` or ``H:MM:SS`` and is converted to
  ``60 * hours + minutes``. A day field and the seconds are discarded, and a
  bare number of minutes is rejected. ``runtime`` (in minutes) is therefore
  written as total hours: 2880 minutes becomes ``48:00:00``.

Resource names follow snakemake-executor-plugin-slurm, so a workflow's
resources work unchanged under either executor.
"""

import re
from typing import Any, List, Optional

from snakemake_interface_common.exceptions import WorkflowError

# A GRES entry, as accepted by snakemake-executor-plugin-slurm:
# name[:type][:count[TGM]], for example "gpu:1" or "gpu:a100:2".
_GRES_RE = re.compile(r"^[a-zA-Z0-9_]+(:[a-zA-Z0-9_.]+)?(:\d+[TGM]?)?$")
_GPU_MODEL_RE = re.compile(r"^[a-zA-Z0-9_.]+$")
_SAFE_VALUE_RE = re.compile(r"^[^\s'\"]+$")
_SLURM_VERSION_RE = re.compile(r"(\d+)\.(\d+)")

# GRES type prefixes that sbatch adds, by Slurm release.
GRES_PREFIXES = ("", "gres:", "gres/")
DEFAULT_GRES_PREFIX = "gres/"

# Options that slurm-drmaa cannot express correctly. They are rejected in
# extra arguments so that GPU requests always go through the 'gres' and 'gpu'
# resources, where the gres/ prefix is added.
_GPU_OPTION_RE = re.compile(
    r"(?:^|\s)(--gpus(?:-per-[a-z]+)?|-G|--cpus-per-gpu|--mem-per-gpu"
    r"|--ntasks-per-gpu|--gpu-bind|--gpu-freq|--gres)(?=[=\s]|$)"
)

# Resources of snakemake-executor-plugin-slurm that slurm-drmaa cannot express.
UNSUPPORTED_RESOURCES = {
    "cpus_per_gpu": "--cpus-per-gpu",
    "tasks_per_gpu": "--ntasks-per-gpu",
}

# Resource names that look meaningful but are not read by this plugin, mapped
# to the name that is. Used for warnings, because an unread key otherwise
# fails silently (the job runs with the cluster default instead).
LEGACY_RESOURCES = {
    "account": "slurm_account",
    "partition": "slurm_partition",
    "time": "runtime (in minutes)",
}


def _is_set(value: Any) -> bool:
    return value is not None and value != ""


def _check_value(name: str, value: Any) -> str:
    text = str(value)
    if not _SAFE_VALUE_RE.match(text):
        raise WorkflowError(
            f"Resource '{name}' has the value {text!r}, which contains "
            "whitespace or quotes. slurm-drmaa splits the native specification "
            "on whitespace and does not interpret quotes, so this value cannot "
            "be passed."
        )
    return text


def _positive_int(name: str, value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise WorkflowError(f"Resource '{name}' must be an integer, not {value!r}.")
    if number < 1:
        raise WorkflowError(f"Resource '{name}' must be at least 1, not {number}.")
    return number


def format_runtime(minutes: Any) -> str:
    """Format a runtime in minutes as total hours, HH:MM:00."""
    total = _positive_int("runtime", minutes)
    return f"{total // 60:02d}:{total % 60:02d}:00"


def gres_prefix_for_slurm_version(version: str) -> Optional[str]:
    """Return the GRES prefix that sbatch adds in the given Slurm release.

    ``version`` may be the output of ``sbatch --version``, such as
    "slurm 23.11.10". Returns None if no version can be read.
    """
    match = _SLURM_VERSION_RE.search(version or "")
    if not match:
        return None
    release = (int(match.group(1)), int(match.group(2)))
    if release >= (23, 11):
        return "gres/"
    if release >= (21, 8):
        return "gres:"
    return ""


def gres_entries(resources, prefix: str = DEFAULT_GRES_PREFIX) -> List[str]:
    """Return the GRES entries for a job, each with the given type prefix."""
    entries = []
    gres = resources.get("gres")
    if _is_set(gres):
        for part in str(gres).split(","):
            part = part.strip()
            for existing in ("gres/", "gres:"):
                if part.startswith(existing):
                    part = part[len(existing) :]
            if not _GRES_RE.match(part):
                raise WorkflowError(
                    f"Invalid 'gres' entry {part!r}. Use name[:type][:count], "
                    "for example 'gpu:1' or 'gpu:a100:2'."
                )
            entries.append(part)

    gpu = resources.get("gpu")
    gpu_model = resources.get("gpu_model")
    if _is_set(gpu):
        if any(entry.split(":")[0] == "gpu" for entry in entries):
            raise WorkflowError(
                "Request GPUs with either the 'gpu' resource or a 'gres' entry, "
                "not both."
            )
        count = _positive_int("gpu", gpu)
        if _is_set(gpu_model):
            model = str(gpu_model)
            if not _GPU_MODEL_RE.match(model):
                raise WorkflowError(f"Invalid 'gpu_model' value {model!r}.")
            entries.append(f"gpu:{model}:{count}")
        else:
            entries.append(f"gpu:{count}")
    elif _is_set(gpu_model):
        raise WorkflowError("'gpu_model' is set, but 'gpu' (the GPU count) is not.")

    return [f"{prefix}{entry}" for entry in entries]


def _check_extra(source: str, text: str) -> str:
    match = _GPU_OPTION_RE.search(text)
    if match:
        raise WorkflowError(
            f"{source} contains '{match.group(1)}'. Request GPUs and other "
            "generic resources with the 'gres' or 'gpu' resource instead. "
            "slurm-drmaa accepts only --gres, and only with the type prefix "
            "that this plugin adds for the cluster's Slurm release."
        )
    if "'" in text or '"' in text:
        raise WorkflowError(
            f"{source} contains quotes. slurm-drmaa does not interpret quotes, "
            "so they would be passed to Slurm literally."
        )
    return text.strip()


def build_native_spec(
    resources,
    threads: Optional[int],
    extra: Optional[str] = None,
    gres_prefix: str = DEFAULT_GRES_PREFIX,
) -> str:
    """Translate a job's resources into a slurm-drmaa native specification.

    ``resources`` is any object with a ``get`` method, such as a Snakemake
    job's ``resources``. ``extra`` holds additional options from the
    executor's ``--slurm-drmaa-args`` setting. ``gres_prefix`` is the GRES
    type prefix for the cluster's Slurm release.
    """
    for name, option in UNSUPPORTED_RESOURCES.items():
        if _is_set(resources.get(name)):
            raise WorkflowError(
                f"Resource '{name}' ({option}) is not supported by slurm-drmaa."
            )

    args = []

    account = resources.get("slurm_account")
    if _is_set(account):
        args.append(f"--account={_check_value('slurm_account', account)}")

    partition = resources.get("slurm_partition")
    if _is_set(partition):
        args.append(f"--partition={_check_value('slurm_partition', partition)}")

    runtime = resources.get("runtime")
    if _is_set(runtime):
        args.append(f"--time={format_runtime(runtime)}")

    mem_per_cpu = resources.get("mem_mb_per_cpu")
    mem = resources.get("mem_mb")
    if _is_set(mem_per_cpu):
        args.append(f"--mem-per-cpu={_positive_int('mem_mb_per_cpu', mem_per_cpu)}")
    elif _is_set(mem):
        args.append(f"--mem={_positive_int('mem_mb', mem)}")

    cpus = resources.get("cpus_per_task")
    if _is_set(cpus):
        args.append(f"--cpus-per-task={_positive_int('cpus_per_task', cpus)}")
    elif threads:
        args.append(f"--cpus-per-task={_positive_int('threads', threads)}")

    for name, option in (
        ("nodes", "--nodes"),
        ("tasks", "--ntasks"),
        ("tasks_per_node", "--ntasks-per-node"),
    ):
        value = resources.get(name)
        if _is_set(value):
            args.append(f"{option}={_positive_int(name, value)}")

    for name, option in (("constraint", "--constraint"), ("qos", "--qos")):
        value = resources.get(name)
        if _is_set(value):
            args.append(f"{option}={_check_value(name, value)}")

    gres = gres_entries(resources, gres_prefix)
    if gres:
        args.append(f"--gres={','.join(gres)}")

    slurm_extra = resources.get("slurm_extra")
    if _is_set(slurm_extra):
        args.append(_check_extra("Resource 'slurm_extra'", str(slurm_extra)))

    if _is_set(extra):
        args.append(_check_extra("--slurm-drmaa-args", extra))

    return " ".join(arg for arg in args if arg)


def legacy_resource_warnings(resources) -> List[str]:
    """Return a warning for each resource name that this plugin does not read."""
    warnings = []
    for legacy, current in LEGACY_RESOURCES.items():
        current_key = current.split()[0]
        if _is_set(resources.get(legacy)) and not _is_set(resources.get(current_key)):
            warnings.append(
                f"Resource '{legacy}' is not read by the slurm-drmaa executor and "
                f"has no effect; use '{current}' instead."
            )
    return warnings
