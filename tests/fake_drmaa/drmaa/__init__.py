"""Stand-in for the drmaa package, for end-to-end tests without a cluster.

runJob records each native specification and runs the jobscript locally.
Environment variables control behaviour:
  FAKE_DRMAA_RECORD  file to which each native specification is appended
  FAKE_DRMAA_REJECT  substring; a specification containing it is rejected
  FAKE_DRMAA_IMPL    value reported as the DRMAA implementation
"""
import itertools
import os
import subprocess

from . import const, errors
from .const import JobState
from .errors import ExitTimeoutException

_ids = itertools.count(1000)


class JobTemplate:
    remoteCommand = nativeSpecification = jobName = None
    outputPath = errorPath = None
    joinFiles = False


class Session:
    _results = {}

    @property
    def drmaaImplementation(self):
        return os.environ.get("FAKE_DRMAA_IMPL", "DRMAA for Slurm 1.1.5")

    def initialize(self):
        pass

    def exit(self):
        pass

    def createJobTemplate(self):
        return JobTemplate()

    def deleteJobTemplate(self, jt):
        pass

    def runJob(self, jt):
        record = os.environ.get("FAKE_DRMAA_RECORD")
        if record:
            with open(record, "a") as fh:
                fh.write(jt.nativeSpecification + "\n")
        reject = os.environ.get("FAKE_DRMAA_REJECT")
        if reject and reject in jt.nativeSpecification:
            raise errors.InternalException(
                "code 1: slurm_submit_batch_job error (-1): Unspecified error"
            )
        jobid = str(next(_ids))
        # Like Slurm: the output path is a file, %j is the job ID, and a
        # directory is not accepted (the job fails without running).
        output = (jt.outputPath or "").lstrip(":").replace("%j", jobid)
        if output and os.path.isdir(output):
            self._results[jobid] = 1
            return jobid
        if output:
            with open(output, "w") as fh:
                self._results[jobid] = subprocess.run(
                    ["bash", jt.remoteCommand], stdout=fh, stderr=subprocess.STDOUT
                ).returncode
        else:
            self._results[jobid] = subprocess.run(["bash", jt.remoteCommand]).returncode
        return jobid

    def jobStatus(self, jobid):
        return JobState.DONE if self._results[jobid] == 0 else JobState.FAILED

    def control(self, jobid, action):
        pass
