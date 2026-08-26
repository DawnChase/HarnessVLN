class HarnessError(RuntimeError):
    """Base error for composition and execution failures."""


class DuplicateToolError(HarnessError):
    pass


class MissingToolError(HarnessError):
    pass


class ToolClosedError(HarnessError):
    pass


class ToolValidationError(HarnessError):
    pass
