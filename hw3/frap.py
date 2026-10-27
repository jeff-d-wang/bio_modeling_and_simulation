import math
import sys

# problem 4
# python frap.py input.txt

def sse(R_inf, k, t, R):
    return sum((Ri - R_inf * (1 - math.exp(-k * ti))) ** 2 for ti, Ri in zip(t, R))


# 4a
def gradient(R_inf, k, t, R):
    dR = dk = 0.0
    for ti, Ri in zip(t, R):
        e = math.exp(-k * ti)
        resid = Ri - R_inf * (1 - e)
        dR += -2 * resid * (1 - e)
        dk += -2 * resid * R_inf * ti * e
    return dR, dk


# 4b
def hessian(R_inf, k, dR, dk, t, R):
    f = lambda a, b: sse(a, b, t, R)
    f00 = f(R_inf, k)
    hRR = (f(R_inf + 2 * dR, k) - 2 * f(R_inf + dR, k) + f00) / dR ** 2
    hkk = (f(R_inf, k + 2 * dk) - 2 * f(R_inf, k + dk) + f00) / dk ** 2
    hRk = (f(R_inf + dR, k + dk) - f(R_inf + dR, k) - f(R_inf, k + dk) + f00) / (dR * dk)
    return hRR, hRk, hkk


# 4c
def newton(R_inf, k, dR, dk, q, t, R):
    for _ in range(q):
        gR, gk = gradient(R_inf, k, t, R)
        hRR, hRk, hkk = hessian(R_inf, k, dR, dk, t, R)
        det = hRR * hkk - hRk ** 2
        R_inf += (-gR * hkk + gk * hRk) / det
        k += (-gk * hRR + gR * hRk) / det
    return R_inf, k


def parse(path):
    inp = open(path).read().split()
    R_inf, k, dR, dk = (float(x) for x in inp[:4])
    q, n = int(inp[4]), int(inp[5])
    values = [float(x) for x in inp[6:6 + 2 * n]]
    # catch errors in the input file
    if len(values) != 2 * n:
        sys.exit(f"{path}: expected {n} data rows")
    return R_inf, k, dR, dk, q, values[0::2], values[1::2]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python frap.py input.txt")
    R_inf, k = newton(*parse(sys.argv[1]))
    print(R_inf, k)
