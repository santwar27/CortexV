"""Block windows shared by durable UI threads and engine short-term memory."""

MIN_PAIRS = 2
DEFAULT_MAX_HISTORY = 10


class HistoryWindow:
    """Keep a stable prefix while the window grows, then discard a block.

    The engine's message budget includes the current exchange: a budget of
    10 historically left 4 prior pairs in the prompt (10 // 2 - 1). Derive
    that maximum here so UI callers and non-default engine budgets agree.
    """

    def __init__(self, max_history: int = DEFAULT_MAX_HISTORY):
        if (not isinstance(max_history, int) or isinstance(max_history, bool)
                or max_history <= 0):
            max_history = DEFAULT_MAX_HISTORY
        self.max_pairs = max(1, max_history // 2 - 1)
        self.min_pairs = min(MIN_PAIRS, self.max_pairs)
        self.step = self.max_pairs - self.min_pairs + 1

    def start(self, n: int) -> int:
        """Absolute first exchange to send, given n complete prior exchanges."""
        return max(0, self.step * ((n - self.min_pairs) // self.step))
