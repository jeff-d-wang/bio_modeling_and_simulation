"""Simulate cell densities for one cell line from the growth model in Problem 1."""
import argparse
import json
import math
from pathlib import Path
import random


def log_curve(t, a, r, t0):
    """Log of the logistic curve K / (1 + exp(-r (t - t0))), with a = log K."""
    z = -r * (t - t0)
    return a - (max(z, 0) + math.log1p(math.exp(-abs(z))))


def draw_parameters(rng):
    """Draw (a, r, t0, sigma) from the priors and return them as a tuple.

    Draw a, log r, t0, and sigma in that order, each with rng.gauss,
    and use the absolute value of the last draw as sigma.
    """
    # BEGIN P1b
    a = rng.gauss(7, 3)
    r = math.exp(rng.gauss(-1.5, 3))
    t0 = rng.gauss(12, 10)
    sigma = abs(rng.gauss(0, 1))
    return a, r, t0, sigma
    # END P1b


def simulate_densities(rng, a, r, t0, sigma, days):
    """Return one simulated density (thousands of cells per mL) for each
    entry of days, using one rng.gauss call per reading.
    """
    # BEGIN P1b
    return [math.exp(rng.gauss(log_curve(t, a, r, t0), sigma)) for t in days]
    # END P1b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--repeats", type=int, default=1, help="readings per day")
    parser.add_argument("--from-prior", action="store_true", help="draw parameters from the priors")
    parser.add_argument("--output", type=Path, default=Path("simulated_growth.json"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    if args.from_prior:
        a, r, t0, sigma = draw_parameters(rng)
    else:
        a, r, t0, sigma = math.log(1200), 0.25, 12.0, args.sigma
    days = [t for t in range(0, 31, 3) for _ in range(args.repeats)]
    densities = simulate_densities(rng, a, r, t0, sigma, days)
    result = {"cell_line": [1] * len(days), "day": days, "density": densities,
              "parameters": {"a": a, "r": r, "t0": t0, "sigma": sigma}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
