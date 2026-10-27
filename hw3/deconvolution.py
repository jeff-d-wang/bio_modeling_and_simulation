"""Fit cell-type amounts to bulk expression for Problem 6. R holds the cell-type
signatures (genes by cell types) and v holds one sample's bulk expression."""
import json
import sys
import cvxpy as cp
import numpy as np


def least_squares(R, v):
    """Return c as a NumPy array minimizing ||Rc - v||^2, from part (a)."""
    # BEGIN P6d
    return np.linalg.solve(R.T @ R, R.T @ v)
    # END P6d


def absolute_fit(R, v, lam):
    """Return c as a NumPy array (c.value) from the linear program in part (b),
    written with its auxiliary variables."""
    # BEGIN P6d
    n, k = R.shape
    c, e, u = cp.Variable(k), cp.Variable(n), cp.Variable(k)
    constraints = [e >= R @ c - v, e >= v - R @ c, u >= c, u >= -c]
    cp.Problem(cp.Minimize(cp.sum(e) + lam * cp.sum(u)), constraints).solve()
    return c.value
    # END P6d


def penalized_fit(R, v, lam):
    """Return c as a NumPy array (c.value) from the quadratic program in part (c),
    written with its auxiliary variables."""
    # BEGIN P6d
    k = R.shape[1]
    c, u = cp.Variable(k), cp.Variable(k)
    constraints = [u >= c, u >= -c]
    cp.Problem(cp.Minimize(cp.sum_squares(R @ c - v) + lam * cp.sum(u)), constraints).solve()
    return c.value
    # END P6d


def report(label, R, v, c, genes):
    residual = R @ c - v
    largest = np.argsort(-np.abs(residual))[:6]
    print(f"{label:30s} c = {np.round(c, 3) + 0.0}  sum = {c.sum():.3f}")
    print(f"{'':30s} largest |residuals|: " +
          ", ".join(f"{genes[i]} {residual[i]:.2f}" for i in largest))


if __name__ == "__main__":
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "PS3-data/deconvolution.json"))
    R, bulk, genes = np.array(d["signatures"]), np.array(d["bulk"]), d["genes"]
    print("Cell types:", ", ".join(d["cell_types"]))
    A, B = bulk[:, 0], bulk[:, 1]
    print("\nSample A")
    report("least squares", R, A, least_squares(R, A), genes)
    for lam in (10, 30, 100):
        report(f"penalized fit, lambda = {lam}", R, A, penalized_fit(R, A, lam), genes)
    print("\nSample B")
    report("least squares", R, B, least_squares(R, B), genes)
    report("absolute-error fit, lambda = 0", R, B, absolute_fit(R, B, 0), genes)
