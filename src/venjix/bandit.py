"""LinUCB contextual bandit (rung 4). Pure Python, no neural nets (per the
baseline hierarchy), no numpy (package stays dependency-free at toy scale).

Per arm a: A_a = I + sum(x x'), b_a = sum(r x); theta_a = A_a^-1 b_a;
select argmax_a  theta_a . x  +  ucb_alpha * sqrt(x . A_a^-1 x).

Deterministic by construction: ties break in fixed arm order and arm selection
uses no RNG, so runs reproduce from config + seed alone.
"""

import math


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for small SPD systems."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = (a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))) / a[row][row]
    return x


class LinUCB:
    def __init__(self, n_arms: int, dim: int, ucb_alpha: float = 1.0):
        if n_arms < 2 or dim < 1 or ucb_alpha < 0:
            raise ValueError("need n_arms >= 2, dim >= 1, ucb_alpha >= 0")
        self.n_arms = n_arms
        self.dim = dim
        self.ucb_alpha = ucb_alpha
        # Ridge prior: A = I, b = 0 per arm.
        self._A = [
            [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
            for _ in range(n_arms)
        ]
        self._b = [[0.0] * dim for _ in range(n_arms)]

    def select(self, x: tuple[float, ...]) -> int:
        assert len(x) == self.dim
        best_arm, best_score = 0, -math.inf
        for arm in range(self.n_arms):
            theta = solve_linear(self._A[arm], self._b[arm])
            a_inv_x = solve_linear(self._A[arm], list(x))
            width = math.sqrt(max(0.0, sum(xi * yi for xi, yi in zip(x, a_inv_x))))
            score = sum(t * xi for t, xi in zip(theta, x)) + self.ucb_alpha * width
            if score > best_score:  # strict: ties keep the earlier arm
                best_arm, best_score = arm, score
        return best_arm

    def update(self, arm: int, x: tuple[float, ...], reward: float) -> None:
        assert len(x) == self.dim
        for i in range(self.dim):
            for j in range(self.dim):
                self._A[arm][i][j] += x[i] * x[j]
            self._b[arm][i] += reward * x[i]
