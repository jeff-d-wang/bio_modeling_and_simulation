"""Problem 1(b): checks and plots for the simulated growth datasets."""
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from simulate_growth import log_curve


def load(path):
    d = json.load(open(path))
    p = d["parameters"]
    return np.array(d["day"]), np.log(d["density"]), p


def plot(ax, path, label=None):
    t, y, p = load(path)
    grid = np.linspace(0, 30, 301)
    line, = ax.plot(grid, [log_curve(s, p["a"], p["r"], p["t0"]) for s in grid], label=label)
    ax.plot(t, y, "o", color=line.get_color())


t, y, p = load("noiseless.json")
gap = max(abs(yj - log_curve(tj, p["a"], p["r"], p["t0"])) for tj, yj in zip(t, y))
print(f"sigma=0 max |y - m(t)| = {gap:.2e}")

t, y, p = load("many_readings.json")
err = y - np.array([log_curve(tj, p["a"], p["r"], p["t0"]) for tj in t])
print(f"repeats=1000 log errors: mean = {err.mean():.4f}, sd = {err.std(ddof=1):.4f}, n = {len(err)}")

fig, ax = plt.subplots()
plot(ax, "simulated_growth.json", "$m(t)$")
ax.set(xlabel="day", ylabel="log density", title="simulations for seed=3101, K=1200, r=0.25, t0=12, sigma=0.12")
fig.savefig("p1b_simulated.png")

fig, ax = plt.subplots()
for i in range(1, 6):
    _, _, p = load(f"prior{i}.json")
    print(f"prior{i}: K = {math.exp(p['a']):.3g}, r = {p['r']:.3g}, t0 = {p['t0']:.3g}, sigma = {p['sigma']:.3g}")
    plot(ax, f"prior{i}.json", f"seed {i}")
ax.set(xlabel="day", ylabel="log density", title="5 datasets given diff priors")
ax.legend()
fig.savefig("p1b_prior.png")
