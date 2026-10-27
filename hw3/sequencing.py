"""Solve the sequencing linear program of Problem 5 with cvxpy and report dual values."""
import json
import cvxpy as cp


def build(tissue, budget, time):
    """Return the problem, the variable x = (bulk, single-cell, spatial), and a Python list of
    three separate resource constraints in the order tissue, budget, and time."""
    x = cp.Variable(3)
    # BEGIN P5d
    # Use the arguments tissue, budget, and time as the resource limits.
    objective = cp.Maximize(x[0] + 2 * x[1] + 3 * x[2])
    resources = [x[0] + 2 * x[1] <= tissue,
                 x[1] + x[2] <= budget,
                 2 * x[0] + x[2] <= time]
    # END P5d
    if objective is None or resources is None:
        raise SystemExit("Complete the P5d block in sequencing.py, then run again.")
    return cp.Problem(objective, resources + [x >= 0]), x, resources


def solve(tissue=3, budget=8, time=10):
    problem, x, resources = build(tissue, budget, time)
    problem.solve()
    return {"supplies": [tissue, budget, time], "x": [round(float(v), 6) for v in x.value],
            "value": round(float(problem.value), 6),
            "dual_values": [round(float(c.dual_value), 6) for c in resources]}


if __name__ == "__main__":
    results = [solve(), solve(tissue=4), solve(budget=9), solve(time=11)]
    print(json.dumps(results, indent=2))
