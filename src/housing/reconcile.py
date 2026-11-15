"""Make forecasts add up: bottom-up and MinTrace reconciliation of node-level forecasts.

Forecasts are reconciled in dollars (total housing-stock value), where the hierarchy holds exactly.
For base forecasts y_hat (all nodes, aggregates first), every method returns

    y_tilde = y_hat - W C' (C W C')^-1 C y_hat,        C = [ I_aggregates  -S_aggregates ]

the coherent forecast closest to y_hat in the metric W^-1 (Wickramasuriya, Athanasopoulos and
Hyndman, 2019). The methods differ only in W, the assumed covariance of base-forecast errors:

* wls_struct  - W = diag(number of counties under each node): ignores scale, uses only the shape of the tree.
* wls_var     - W = diag(past error variance of each node, in $ at today's level).
* mint_shrink - full W from past errors, correlations shrunk towards zero (Schafer-Strimmer).
                With 2,000+ nodes and a few dozen past errors, W is never formed: it is kept as
                diagonal + low rank, and C W C' (one row per aggregate) is built from those pieces.

Past errors come only from earlier origins whose outcome was already known at the current origin,
so the weights are time-safe.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


class Hierarchy:
    def __init__(self, S: np.ndarray, n_agg: int):
        self.S = S                      # m x n_bottom, aggregates first, bottom rows = identity
        self.n_agg = n_agg
        self.Sa = sparse.csr_matrix(S[:n_agg])   # each county has only 3 parents: very sparse
        self.struct = S.sum(1)          # counties under each node

    def C(self, v: np.ndarray) -> np.ndarray:
        """C @ v for v of shape (m,) or (m, k): aggregate minus the sum of its counties."""
        return v[: self.n_agg] - self.Sa @ v[self.n_agg:]

    def Ct(self, u: np.ndarray) -> np.ndarray:
        """C' @ u."""
        return np.concatenate([u, -self.Sa.T @ u])

    def _SwS(self, w: np.ndarray) -> np.ndarray:
        """S_a diag(w) S_a' as a dense n_agg x n_agg array."""
        return (self.Sa.multiply(w[None, :]) @ self.Sa.T).toarray()

    def bottom_up(self, yhat: np.ndarray) -> np.ndarray:
        return self.S @ yhat[self.n_agg:]

    def wls(self, yhat: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Diagonal W = diag(w)."""
        M = np.diag(w[: self.n_agg]) + self._SwS(w[self.n_agg:])
        return yhat - w * self.Ct(np.linalg.solve(M, self.C(yhat)))

    def mint_lowrank(self, yhat: np.ndarray, d: np.ndarray, Z: np.ndarray, lam: float) -> np.ndarray:
        """W = Dd (lam I + (1-lam)/n Z'Z) Dd, with Dd = diag(d), Z = n x m standardised past errors."""
        n = Z.shape[0]
        d2 = d ** 2
        A = self.C((d[:, None] * Z.T))                               # n_agg x n  (= C Dd Z')
        M = lam * (np.diag(d2[: self.n_agg]) + self._SwS(d2[self.n_agg:])) + (1 - lam) / n * A @ A.T
        u = np.linalg.solve(M, self.C(yhat))
        v = self.Ct(u)
        Wv = lam * d2 * v + (1 - lam) / n * d * (Z.T @ (Z @ (d * v)))
        return yhat - Wv


def shrinkage_lambda(Z: np.ndarray) -> float:
    """Schafer-Strimmer shrinkage intensity for the correlation matrix of standardised errors Z (n x m).

    Computed from n x n Gram matrices, never forming the m x m correlation matrix.
    """
    n = Z.shape[0]
    if n < 3:
        return 1.0
    W2 = Z ** 2
    diag_wbar = W2.mean(0)                                    # w_bar_ii
    G = Z @ Z.T                                               # n x n
    sum_wbar2_all = (G ** 2).sum() / n ** 2                   # sum_ij w_bar_ij^2
    sum_wbar2_off = sum_wbar2_all - (diag_wbar ** 2).sum()
    sum_w2_off = ((W2.sum(1)) ** 2 - (W2 ** 2).sum(1)).sum()  # sum_k sum_{i!=j} w_kij^2
    var_off = n / (n - 1) ** 3 * (sum_w2_off - n * sum_wbar2_off)
    denom = (n / (n - 1)) ** 2 * sum_wbar2_off
    return float(np.clip(var_off / denom, 0.0, 1.0)) if denom > 0 else 1.0


def standardise(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Past log errors E (n x m) -> (Z standardised, sd). Nodes with no spread get sd = median sd."""
    sd = E.std(0, ddof=1)
    good = np.isfinite(sd) & (sd > 0)
    sd = np.where(good, sd, np.nanmedian(sd[good]) if good.any() else 1.0)
    Z = (E - E.mean(0)) / sd
    return np.nan_to_num(Z), sd
