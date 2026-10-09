import numpy as np
import matplotlib.pyplot as plt

S_1 = 1
S_2 = 3

def p_density(p, s, m=0):

    eta = np.log(p / (1 - p))
    
    derivative = 1 / (p * (1 - p))
    
    normal_pdf = (1 / np.sqrt(2 * np.pi * s**2)) * np.exp(-((eta - m)**2) / (2 * s**2))
    
    return normal_pdf * derivative

p_linear = np.linspace(1e-5, 1 - 1e-5, 5000)
p_log = np.logspace(-5, np.log10(0.5), 1000)

plt.figure(figsize=(7, 5))

plt.plot(p_linear, p_density(p_linear, s=S_1), label=f's = {S_1}', color='blue')
plt.plot(p_linear, p_density(p_linear, s=S_2), label=f's = {S_2}', color='red')
plt.title('Density of p (Linear Scale)')
plt.xlabel('Probability (p)')
plt.ylabel('Density g(p)')
plt.ylim(0, 5)
plt.legend()

plt.tight_layout()
plt.savefig("p6c.png", dpi=300, bbox_inches="tight")