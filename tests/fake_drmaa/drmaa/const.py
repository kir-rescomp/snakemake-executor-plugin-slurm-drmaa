class JobState:
    DONE = "done"
    FAILED = "failed"
    RUNNING = "running"
    USER_SUSPENDED = "user_suspended"
    SYSTEM_SUSPENDED = "system_suspended"


class JobControlAction:
    TERMINATE = "terminate"
