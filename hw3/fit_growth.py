"""Fit a growth model with CmdStanPy and save posterior summaries and draws."""
# Runs (from hw3/):
#   1c: python fit_growth.py growth_base.stan simulated_growth.json --output base_fit
#   1d: python fit_growth.py growth_extend.stan PS3-data/growth.json --output extended_fit
#   1e: python fit_growth.py growth_base.stan line12.json --output line12_base_fit
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from cmdstanpy import CmdStanModel


def summarize(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("growth_output"))
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()
    d = json.loads(args.data.read_text())
    cell_line, day, density = (np.asarray(d[k]) for k in ("cell_line", "day", "density"))
    if len(day) == 0 or np.any(density <= 0):
        parser.error("Data must contain positive densities and at least one observation")
    labels = sorted(set(cell_line.tolist()))
    ids = {label: i + 1 for i, label in enumerate(labels)}
    available = {"N": len(day), "M": len(labels), "cell_line": [ids[int(i)] for i in cell_line],
                 "t": day.tolist(), "y": np.log(density).tolist()}
    model = CmdStanModel(stan_file=str(args.model.resolve()))
    data = {name: available[name] for name in model.src_info()["inputs"]}

    # Starting values near typical curves, different for each of the four chains.
    parameters = model.src_info()["parameters"]
    starts = []
    for chain in range(4):
        a0, logr0 = 6.8 + .2 * chain, -1.7 + .1 * chain
        start = {"t0": 10.0 + chain, "sigma": .10 + .03 * chain,
                 "mu_a": a0, "mu_logr": logr0, "tau_a": .3, "tau_logr": .3}
        for name, value in (("a", a0), ("logr", logr0)):
            start[name] = [value] * len(labels) if parameters.get(name, {}).get("dimensions") else value
        starts.append({k: v for k, v in start.items() if k in parameters})

    args.output.mkdir(parents=True, exist_ok=True)
    fit = model.sample(data=data, chains=4, parallel_chains=4, seed=args.seed, inits=starts,
                       adapt_delta=.95, show_progress=False, output_dir=str(args.output / "chains"))
    draws = fit.stan_variables()
    np.savez_compressed(args.output / "draws.npz", **draws)
    rows = []
    for name, values in draws.items():
        flat = values.reshape(len(values), -1)
        for j in range(flat.shape[1]):
            if values.ndim == 1:
                label = name
            elif flat.shape[1] == len(labels):
                label = f"{name}[{labels[j]}]"
            else:
                label = f"{name}[{j + 1}]"
            rows.append({"parameter": label, **summarize(flat[:, j])})
            if name in {"a", "logr", "mu_a", "mu_logr"}:
                rows.append({"parameter": f"exp({label})", **summarize(np.exp(flat[:, j]))})
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "mean", "sd", "q05", "q95"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary.csv and draws.npz to {args.output}")


if __name__ == "__main__":
    main()
