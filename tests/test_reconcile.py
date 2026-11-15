import numpy as np
import pytest

from housing.reconcile import Hierarchy, shrinkage_lambda, standardise


@pytest.fixture
def H():
    # US > {A, B}; grouping 2: M1 = {a1, b1}, M2 = {a2}; bottom a1, a2, b1
    Sa = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]], float)
    return Hierarchy(np.vstack([Sa, np.eye(3)]), n_agg=5)


def test_all_methods_are_coherent(H):
    rng = np.random.default_rng(1)
    yhat = rng.uniform(10, 20, 8)
    E = rng.normal(size=(20, 8))
    Z, sd = standardise(E)
    for y in (H.bottom_up(yhat), H.wls(yhat, H.struct), H.wls(yhat, sd ** 2),
              H.mint_lowrank(yhat, sd, Z, shrinkage_lambda(Z))):
        assert np.allclose(H.C(y), 0)


def test_coherent_input_is_left_unchanged(H):
    y = H.S @ np.array([3.0, 4.0, 5.0])
    assert np.allclose(H.wls(y, H.struct), y)
    Z, sd = standardise(np.random.default_rng(0).normal(size=(10, 8)))
    assert np.allclose(H.mint_lowrank(y, sd, Z, 0.3), y)


def test_lowrank_mint_equals_textbook_formula(H):
    """y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat with the full W."""
    rng = np.random.default_rng(2)
    yhat = rng.uniform(10, 20, 8)
    Z, sd = standardise(rng.normal(size=(12, 8)))
    lam = 0.4
    W = np.diag(sd) @ (lam * np.eye(8) + (1 - lam) / 12 * Z.T @ Z) @ np.diag(sd)
    Wi = np.linalg.inv(W)
    S = H.S
    expected = S @ np.linalg.solve(S.T @ Wi @ S, S.T @ Wi @ yhat)
    assert np.allclose(H.mint_lowrank(yhat, sd, Z, lam), expected)


def test_wls_struct_matches_hierarchicalforecast(H):
    from hierarchicalforecast.methods import MinTrace

    rng = np.random.default_rng(3)
    yhat = rng.uniform(10, 20, (8, 1))
    tags = {"all": np.arange(8)}
    lib = MinTrace(method="wls_struct").fit_predict(S=H.S, y_hat=yhat, tags=tags)
    assert np.allclose(H.wls(yhat[:, 0], H.struct), np.asarray(lib["mean"])[:, 0])


def test_shrinkage_lambda_matches_brute_force():
    rng = np.random.default_rng(4)
    Z, _ = standardise(rng.normal(size=(15, 6)) @ rng.normal(size=(6, 6)))
    n = Z.shape[0]
    w = np.einsum("ki,kj->kij", Z, Z)
    wbar = w.mean(0)
    off = ~np.eye(6, dtype=bool)
    var_r = n / (n - 1) ** 3 * ((w - wbar) ** 2).sum(0)
    r = n / (n - 1) * wbar
    expected = np.clip(var_r[off].sum() / (r[off] ** 2).sum(), 0, 1)
    assert np.isclose(shrinkage_lambda(Z), expected)
