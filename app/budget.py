from app.config import settings
from app.errors import BudgetExceeded


class Budget:
    """Per-query cap: max steps (planner calls + tool calls) and token budget."""

    def __init__(self) -> None:
        self.steps = 0
        self.tool_calls = 0
        self.tokens = 0

    def step(self, tool: bool = False) -> None:
        self.steps += 1
        if tool:
            self.tool_calls += 1
        if self.steps > settings.max_steps:
            raise BudgetExceeded(f"step cap of {settings.max_steps} reached")

    def add_tokens(self, n: int) -> None:
        self.tokens += int(n)
        if self.tokens > settings.token_budget:
            raise BudgetExceeded(f"token budget of {settings.token_budget} reached")
