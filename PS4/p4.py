import argparse

import matplotlib.pyplot as plt
import numpy as np

# problem 4e
# python p4.py --a 1 --b 0.25 --c 0.1 --d 1 --x0 50 --y0 1 --T 100 --dt 0.01


def f(x, y, a, b, c, d):
    return a * x - b * x * y, c * x * y - d * y


def forward_euler(a, b, c, d, x0, y0, T, dt):
    n = int(round(T / dt))
    x, y = np.empty(n + 1), np.empty(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        fx, fy = f(x[i], y[i], a, b, c, d)
        x[i + 1] = x[i] + dt * fx
        y[i + 1] = y[i] + dt * fy
    return x, y


def newton_step(xn, yn, a, b, c, d, dt, rounds=10):
    # 4c: solve u - xn - dt*fx(u,v) = 0, v - yn - dt*fy(u,v) = 0
    u, v = xn, yn
    for _ in range(rounds):
        # eval F(u, v) = [F1, F2]^T
        F1 = u - xn - dt * (a * u - b * u * v)
        F2 = v - yn - dt * (c * u * v - d * v)
        # eval Jacobian J(u, v) of residual system
        J11 = 1.0 - dt * (a - b * v)
        J12 = dt * b * u
        J21 = -dt * c * v
        J22 = 1.0 - dt * (c * u - d)
        # eval det(J)
        detJ = J11 * J22 - J12 * J21
        du = (J22 * F1 - J12 * F2) / detJ
        dv = (-J21 * F1 + J11 * F2) / detJ
        # update
        u -= du
        v -= dv
    return u, v


def backward_euler(a, b, c, d, x0, y0, T, dt):
    n = int(round(T / dt))
    x, y = np.empty(n + 1), np.empty(n + 1)
    x[0], y[0] = x0, y0
    for i in range(n):
        x[i + 1], y[i + 1] = newton_step(x[i], y[i], a, b, c, d, dt)
    return x, y


def leapfrog(a, b, c, d, x0, y0, T, dt):
    n = int(round(T / dt))
    # intialize
    x, y = np.empty(n + 1), np.empty(n + 1)
    x[0], y[0] = x0, y0
    # one forward Euler step to start
    fx, fy = f(x0, y0, a, b, c, d)
    x[1], y[1] = x0 + dt * fx, y0 + dt * fy
    for i in range(1, n):
        fx, fy = f(x[i], y[i], a, b, c, d)
        x[i + 1] = x[i - 1] + 2 * dt * fx
        y[i + 1] = y[i - 1] + 2 * dt * fy
    return x, y


def main():
    p = argparse.ArgumentParser()
    for name, default in [("a", 1), ("b", 0.25), ("c", 0.1), ("d", 1),
                          ("x0", 50), ("y0", 1), ("T", 100), ("dt", 0.01)]:
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--out", default="lv")
    args = p.parse_args()
    params = (args.a, args.b, args.c, args.d, args.x0, args.y0, args.T, args.dt)

    t = np.linspace(0, args.T, int(round(args.T / args.dt)) + 1)
    for label, method in [("Forward Euler", forward_euler),
                          ("Backward Euler", backward_euler),
                          ("Leapfrog", leapfrog)]:
        x, y = method(*params)
        # stop plotting once the solution blows up, so the spike doesn't hide everything else
        bad = ~np.isfinite(x) | ~np.isfinite(y) | (np.abs(x) > 1e4) | (np.abs(y) > 1e4)
        k = np.argmax(bad) if bad.any() else len(t)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(t[:k], x[:k], label="$x$ (lantern flies)")
        ax.plot(t[:k], y[:k], label="$y$ (praying mantises)")
        ax.set_xlim(0, args.T)
        ax.set_xlabel("time")
        ax.set_ylabel("population")
        title = f"{label}; $\\Delta t$ = {args.dt:g}"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        path = f"{args.out}_{method.__name__}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"{label:15s} x(T) = {x[-1]:.4g}, y(T) = {y[-1]:.4g} -> {path}")


if __name__ == "__main__":
    main()
