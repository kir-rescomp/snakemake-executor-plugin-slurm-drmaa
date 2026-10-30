class DrmaaException(Exception):
    pass


class AlreadyActiveSessionException(DrmaaException):
    pass


class ExitTimeoutException(DrmaaException):
    pass


class InternalException(DrmaaException):
    pass


class InvalidJobException(DrmaaException):
    pass


class DeniedByDrmException(DrmaaException):
    pass
