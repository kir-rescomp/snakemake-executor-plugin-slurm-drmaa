"""Snakemake executor plugin: submit jobs to Slurm through slurm-drmaa.

Forked from snakemake-executor-plugin-drmaa. Instead of a single hand-written
native specification, job options are built from each rule's resources, using
the resource names of snakemake-executor-plugin-slurm. See native_spec.py for
the slurm-drmaa behaviours this accounts for, including GPU requests.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Generator, List, Optional

from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.jobs import JobExecutorInterface
from snakemake_interface_executor_plugins.settings import (
    CommonSettings,
    ExecutorSettingsBase,
)

from .native_spec import (
    DEFAULT_GRES_PREFIX,
    build_native_spec,
    gres_prefix_for_slurm_version,
    legacy_resource_warnings,
)


DEFAULT_LOG_DIR = ".snakemake/slurm_drmaa_logs"


@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    args: Optional[str] = field(
        default=None,
        metadata={
            "help": "Additional slurm-drmaa options, appended after the options "
            "built from each rule's resources. Wildcards and job properties may "
            "be used, as in '--comment={rule}'. GPU options are not accepted "
            "here; use the 'gres' or 'gpu' resource."
        },
    )
    log_dir: Optional[Path] = field(
        default=None,
        metadata={
            "help": "Directory for job logs. Each job's standard output and error "
            "are written to <log dir>/<rule>/<Slurm job ID>.log. Relative paths "
            f"are taken from the invocation directory. Default: {DEFAULT_LOG_DIR}."
        },
    )
    gres_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "GRES type prefix to add to each --gres entry: 'gres/' "
            "(Slurm 23.11 and later), 'gres:' (21.08 to 23.02) or 'none' (20.11 "
            "and earlier). By default it is chosen from 'sbatch --version'."
        },
    )
    no_diagnose: bool = field(
        default=False,
        metadata={
            "help": "Do not run 'sbatch --test-only' when a submission fails. By "
            "default the plugin runs it with the same options to report Slurm's "
            "own rejection message, which the DRMAA library does not provide."
        },
    )


common_settings = CommonSettings(
    non_local_exec=True,
    implies_no_shared_fs=False,
    job_deploy_sources=False,
    pass_default_storage_provider_args=True,
    pass_default_resources_args=True,
    pass_envvar_declarations_to_cmd=True,
    auto_deploy_default_storage_provider=False,
)


class Executor(RemoteExecutor):
    def __post_init__(self):
        try:
            import drmaa
            from drmaa.errors import AlreadyActiveSessionException
        except RuntimeError as e:
            raise WorkflowError(f"Error loading DRMAA support:\n{e}")
        self.session = drmaa.Session()
        settings = self.workflow.executor_settings
        self.extra_args = settings.args
        # Slurm writes job output to a file path and does not accept a
        # directory, so each job gets its own file inside this directory.
        self.log_dir = Path(settings.log_dir or DEFAULT_LOG_DIR).resolve()
        self.diagnose = not settings.no_diagnose
        self.suspended_msg = set()
        self.warned_rules = set()
        try:
            self.session.initialize()
        except AlreadyActiveSessionException:
            pass
        self.gres_prefix = self._resolve_gres_prefix(settings.gres_prefix)
        implementation = self.session.drmaaImplementation
        if "slurm" not in str(implementation).lower():
            raise WorkflowError(
                "The slurm-drmaa executor requires the Slurm DRMAA library, but "
                f"the loaded DRMAA implementation is '{implementation}'. Check "
                "DRMAA_LIBRARY_PATH, or use the generic drmaa executor."
            )

    def run_job(self, job: JobExecutorInterface):
        import drmaa

        self._warn_legacy_resources(job)
        try:
            extra = job.format_wildcards(self.extra_args) if self.extra_args else None
            native_spec = build_native_spec(
                job.resources, job.threads, extra=extra, gres_prefix=self.gres_prefix
            )
        except (AttributeError, WorkflowError) as e:
            raise WorkflowError(f"Cannot submit job '{job.name}': {e}")

        jobscript = self.get_jobscript(job)
        self.write_jobscript(job, jobscript)
        rule_log_dir = self.log_dir / job.name
        rule_log_dir.mkdir(parents=True, exist_ok=True)
        # Slurm replaces %j with the job ID when it opens the file.
        log_pattern = str(rule_log_dir / "%j.log")

        job_info = SubmittedJobInfo(job, aux={"jobscript": jobscript})
        jt = self.session.createJobTemplate()
        try:
            jt.remoteCommand = jobscript
            jt.nativeSpecification = native_spec
            jt.outputPath = f":{log_pattern}"
            jt.joinFiles = True
            jt.jobName = os.path.basename(jobscript)
            job_info.external_jobid = self.session.runJob(jt)
            job_info.aux["log"] = log_pattern.replace("%j", str(job_info.external_jobid))
        except drmaa.errors.DrmaaException as e:
            msg = (
                f"Error submitting job (DRMAA error {e}).\n"
                f"Native specification: {native_spec}"
            )
            diagnosis = self._diagnose(native_spec)
            if diagnosis:
                msg += f"\n{diagnosis}"
            # Snakemake appends its own sentence directly after the message.
            self.report_job_error(job_info, msg=msg + "\n")
            return
        finally:
            self.session.deleteJobTemplate(jt)

        self.logger.info(
            f"Submitted DRMAA job {job.jobid} with external jobid "
            f"{job_info.external_jobid} (log: {job_info.aux['log']})."
        )
        self.logger.debug(f"Native specification: {native_spec}")
        self.report_job_submission(job_info)

    def _resolve_gres_prefix(self, setting: Optional[str]) -> str:
        """Choose the GRES type prefix for this cluster's Slurm release."""
        if setting is not None:
            prefixes = {"none": "", "gres:": "gres:", "gres/": "gres/"}
            if setting not in prefixes:
                raise WorkflowError(
                    f"Invalid --slurm-drmaa-gres-prefix '{setting}'; use "
                    "'gres/', 'gres:' or 'none'."
                )
            return prefixes[setting]
        version = None
        sbatch = shutil.which("sbatch")
        if sbatch:
            try:
                result = subprocess.run(
                    [sbatch, "--version"], capture_output=True, text=True, timeout=30
                )
                version = result.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        prefix = gres_prefix_for_slurm_version(version)
        if prefix is None:
            self.logger.warning(
                "Could not determine the Slurm release from 'sbatch --version'; "
                f"assuming the GRES prefix '{DEFAULT_GRES_PREFIX}' (Slurm 23.11 and "
                "later). Set --slurm-drmaa-gres-prefix if GPU jobs are rejected."
            )
            return DEFAULT_GRES_PREFIX
        self.logger.debug(f"Slurm release '{version}': GRES prefix '{prefix}'.")
        return prefix

    def _warn_legacy_resources(self, job: JobExecutorInterface):
        if job.name in self.warned_rules:
            return
        self.warned_rules.add(job.name)
        for warning in legacy_resource_warnings(job.resources):
            self.logger.warning(f"Job '{job.name}': {warning}")

    def _diagnose(self, native_spec: str) -> Optional[str]:
        """Ask Slurm directly why it would reject these options.

        slurm-drmaa reports most rejections as 'Unspecified error'. Running
        'sbatch --test-only' with the same options validates the request
        without submitting it and prints Slurm's own message.
        """
        if not self.diagnose:
            return None
        sbatch = shutil.which("sbatch")
        if sbatch is None:
            return None
        cmd = [sbatch, "--test-only", *native_spec.split(), "--wrap=true"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        output = (result.stderr or result.stdout).strip()
        command = shlex.join(["sbatch", "--test-only", *native_spec.split(), "--wrap=true"])
        if result.returncode != 0:
            return f"Slurm reports (from '{command}'):\n  {output}"
        return (
            f"'{command}' accepts these options, so the rejection is specific "
            "to the DRMAA submission path."
        )

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> Generator[SubmittedJobInfo, None, None]:
        import drmaa

        for active_job in active_jobs:
            jobscript = active_job.aux["jobscript"]
            async with self.status_rate_limiter:
                try:
                    retval = self.session.jobStatus(active_job.external_jobid)
                except drmaa.ExitTimeoutException:
                    yield active_job
                    continue
                except Exception as e:
                    self.logger.error(f"DRMAA Error: {e}")
                    os.remove(jobscript)
                    self.report_job_error(active_job)
                    continue
                if retval == drmaa.JobState.DONE:
                    os.remove(jobscript)
                    self.report_job_success(active_job)
                elif retval == drmaa.JobState.FAILED:
                    os.remove(jobscript)
                    self.report_job_error(active_job, msg=self._failure_message(active_job))
                else:
                    yield active_job
                    if retval == drmaa.JobState.USER_SUSPENDED:
                        self._handle_suspended(active_job, "user")
                    elif retval == drmaa.JobState.SYSTEM_SUSPENDED:
                        self._handle_suspended(active_job, "system")
                    else:
                        self.suspended_msg.discard(active_job.job.jobid)

    def _failure_message(self, job_info: SubmittedJobInfo) -> str:
        jobid = job_info.external_jobid
        # Snakemake appends its own sentence directly after the message.
        return (
            f"Slurm job {jobid} failed.\n"
            f"Job log: {job_info.aux.get('log', 'unknown')}\n"
            f"Slurm's record of the job: sacct -j {jobid} "
            "--format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList\n"
        )

    def _handle_suspended(self, active_job: SubmittedJobInfo, by: str):
        if active_job.job.jobid not in self.suspended_msg:
            self.logger.warning(
                f"Job {active_job.job.jobid} (DRMAA id: "
                f"{active_job.external_jobid}) was suspended by {by}."
            )
            self.suspended_msg.add(active_job.job.jobid)

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        from drmaa.const import JobControlAction
        from drmaa.errors import InternalException, InvalidJobException

        for job_info in active_jobs:
            try:
                self.session.control(
                    job_info.external_jobid, JobControlAction.TERMINATE
                )
            except (InvalidJobException, InternalException):
                pass

    def shutdown(self):
        super().shutdown()
        self.session.exit()

    def get_job_exec_prefix(self, job: JobExecutorInterface):
        if self.workflow.storage_settings.assume_common_workdir:
            return f"cd {shlex.quote(self.workflow.workdir_init)}"
        return ""
