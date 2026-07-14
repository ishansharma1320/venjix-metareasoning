"""THE arbitration signal (Design decision 2).

The threshold heuristic and the contextual-bandit arbiter must consume the
IDENTICAL EWMA of the binary misprediction rate — both import this class and
nothing else. Initialized at 0.0: a cold start is not an emergency; the world
model earns distrust through mispredictions.
"""


class EwmaPredictionError:
    def __init__(self, alpha: float):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value

    def update(self, mispredicted: bool) -> float:
        self._value = self.alpha * float(mispredicted) + (1.0 - self.alpha) * self._value
        return self._value
