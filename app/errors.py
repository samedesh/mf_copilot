class ToolError(Exception):
    """A tool failed in a way the agent can see and react to."""


class BudgetExceeded(Exception):
    """Step cap or token budget hit."""


class LLMError(Exception):
    """Endpoint failure or unrecoverable malformed output."""
