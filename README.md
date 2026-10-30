# snakemake-executor-plugin-slurm-drmaa

A [Snakemake](https://snakemake.github.io) executor plugin that submits jobs to Slurm through the Slurm DRMAA library ([slurm-drmaa](https://github.com/natefoo/slurm-drmaa)). Job options are built from each rule's `resources`, using the same resource names as [snakemake-executor-plugin-slurm](https://github.com/snakemake/snakemake-executor-plugin-slurm). The same workflow therefore runs under either executor without changes, and GPU jobs can be submitted through DRMAA.

This plugin is a fork of [snakemake-executor-plugin-drmaa](https://github.com/snakemake/snakemake-executor-plugin-drmaa).

## Why this fork exists

The upstream DRMAA plugin is scheduler-neutral. It passes one hand-written native specification, given with `--drmaa-args`, to every job. This has three consequences on Slurm:

- **Resource names are not interpreted.** Rules must expose whatever names the specification string references, and a missing name fails the submission.
- **Options cannot be omitted per rule.** A GPU option in the shared string applies to every job, so CPU rules need placeholder values.
- **GPU submission fails on Slurm 21.08 and later.** `slurm-drmaa` accepts only `--gres` for GPUs and copies its value into the job request unchanged. `sbatch` adds a type prefix to each entry first; without it, Slurm rejects the request or does not count it as a GPU. DRMAA reports this only as `Unspecified error` or `Invalid Trackable RESource (TRES) specification`. This is [slurm-drmaa issue #68](https://github.com/natefoo/slurm-drmaa/issues/68).

This plugin builds the options for each job separately, includes an option only when its resource is set, and writes GPU requests in the form Slurm recognises.

## Installation

```bash
pip install git+https://github.com/kir-rescomp/snakemake-executor-plugin-slurm-drmaa
```

The plugin requires the `slurm-drmaa` library. Set `DRMAA_LIBRARY_PATH` to its location, for example `/usr/lib64/libdrmaa.so`. At start-up the plugin checks that the loaded DRMAA implementation is the Slurm one.

It installs alongside the upstream `drmaa` executor without conflict.

## Usage

```bash
snakemake --executor slurm-drmaa --jobs 20 --slurm-drmaa-log-dir logs
```

Resources are set per rule, or for all rules with `--default-resources` and `--set-resources`:

```python
rule segment:
    threads: 16
    resources:
        slurm_account="gpu_kir.prj",
        slurm_partition="gpu_interactive",
        runtime=240,
        mem_mb=96000,
        gpu=1,
    shell: "..."
```

On Slurm 23.11 or later, this rule is submitted with:

```
--account=gpu_kir.prj --partition=gpu_interactive --time=04:00:00 --mem=96000 --cpus-per-task=16 --gres=gres/gpu:1
```

## Resources

| Resource | Option | Notes |
|---|---|---|
| `slurm_account` | `--account` | |
| `slurm_partition` | `--partition` | |
| `runtime` | `--time` | In minutes. Written as total hours; see [Time limits](#time-limits). |
| `mem_mb` | `--mem` | Ignored if `mem_mb_per_cpu` is set. |
| `mem_mb_per_cpu` | `--mem-per-cpu` | |
| `threads` (rule directive) | `--cpus-per-task` | Overridden by the `cpus_per_task` resource. |
| `cpus_per_task` | `--cpus-per-task` | |
| `nodes` | `--nodes` | |
| `tasks` | `--ntasks` | |
| `tasks_per_node` | `--ntasks-per-node` | |
| `constraint` | `--constraint` | |
| `qos` | `--qos` | |
| `gres` | `--gres` | For example `gpu:1`, `gpu:a100:2` or `gpu:1,shard:1`. |
| `gpu`, `gpu_model` | `--gres` | `gpu=2, gpu_model="l4"` becomes `--gres=gres/gpu:l4:2` on Slurm 23.11 or later. |
| `slurm_extra` | appended verbatim | Must not contain GPU options. |

A resource that is unset, or set to `None` or an empty string, adds no option.

The resources `cpus_per_gpu` and `tasks_per_gpu` are rejected, because `slurm-drmaa` cannot express them.

If a rule sets `account`, `partition` or `time` without the corresponding name above, the plugin logs a warning. These names have no effect, and the job would otherwise run silently with the cluster defaults.

## GPU requests

Request GPUs with either the `gpu` resource (with an optional `gpu_model`) or a `gpu` entry in `gres`, not both.

The plugin adds to every `--gres` entry the type prefix that `sbatch` itself adds in the cluster's Slurm release:

| Slurm release | Prefix | Example |
|---|---|---|
| 20.11 and earlier | none | `--gres=gpu:1` |
| 21.08 to 23.02 | `gres:` | `--gres=gres:gpu:1` |
| 23.11 and later | `gres/` | `--gres=gres/gpu:1` |

The release is read from `sbatch --version` at start-up. If it cannot be read, the plugin logs a warning and assumes 23.11 or later. Use `--slurm-drmaa-gres-prefix` (`gres/`, `gres:` or `none`) to set the prefix explicitly. Entries written with either prefix are normalised, so `gres="gres:gpu:1"` and `gres="gpu:1"` give the same result.

`slurm-drmaa` rejects `--gpus`, `--gpus-per-node`, `--cpus-per-gpu`, `--mem-per-gpu` and related options. The plugin also refuses these, and `--gres`, in `slurm_extra` and `--slurm-drmaa-args`, so that every GPU request receives the correct prefix.

## Time limits

`slurm-drmaa` reads the time limit as `H:MM` or `H:MM:SS` and computes 60 × hours + minutes. It discards a day field and the seconds, so the Slurm form `1-00:00:00` would become zero minutes, and it rejects a bare number of minutes. The plugin therefore writes `runtime` as total hours: 2880 minutes becomes `48:00:00`. The parser places no upper limit on the hours.

## Additional options

`--slurm-drmaa-args` appends options to every job, after those built from resources. Job properties and wildcards may be used, for example `--slurm-drmaa-args "--comment={rule}"`.

`slurm-drmaa` splits the native specification on spaces and tabs and does not interpret quotes. Resource values and extra options therefore must not contain whitespace or quotes; the plugin rejects them with an explanatory message.

## Diagnosing rejected submissions

When a submission fails, the error message includes the native specification that was sent. The plugin also runs `sbatch --test-only` with the same options, which validates the request without submitting it, and includes Slurm's own message. For example, a GPU partition's minimum-GPU policy appears as `QOSMinGRES` rather than `Unspecified error`. Use `--slurm-drmaa-no-diagnose` to turn this off.

## Settings

| Setting | Description |
|---|---|
| `--slurm-drmaa-args` | Additional options appended to every job. |
| `--slurm-drmaa-log-dir` | Directory for job stdout and stderr files. Defaults to the working directory. |
| `--slurm-drmaa-gres-prefix` | GRES type prefix: `gres/`, `gres:` or `none`. Detected from `sbatch --version` by default. |
| `--slurm-drmaa-no-diagnose` | Do not run `sbatch --test-only` after a failed submission. |

## Limitations

- Only options in the `slurm-drmaa` subset are available. Others can be passed through `slurm_extra` if `slurm-drmaa` supports them; unsupported options are reported as `Unsupported option`.
- Job status comes from DRMAA, which reports success or failure but not Slurm's reason for a failure. Check the job's log files, or `sacct -j <jobid>`.

## Testing

```bash
pip install -e . pytest snakemake
pytest
```

The end-to-end tests run Snakemake with this executor against a stand-in `drmaa` module in `tests/fake_drmaa`, so no cluster is needed.

## Licence

MIT. See [LICENSE](LICENSE).
