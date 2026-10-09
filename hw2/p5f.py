import numpy as np

N_STEPS = 20
N_SAMPLES = 10**5

def bisection_sampler(n_samples):
    u = np.random.uniform(0, 1, n_samples)
    left = np.zeros(n_samples)
    right = np.ones(n_samples)
    
    for _ in range(N_STEPS):
        m = (left + right) / 2.0
        F_m = (1.0 / np.pi) * np.sin(np.pi * m) - m * np.cos(np.pi * m)
        
        left = np.where(F_m < u, m, left)
        right = np.where(F_m >= u, m, right)
        
    return (left + right) / 2.0

def rejection_sampler(n_samples):

    rejection_samples = np.empty(n_samples)
    proposals_count = 0
    accepted_count = 0
    
    while accepted_count < n_samples:
        batch_size = n_samples - accepted_count
        
        x_prop = np.sqrt(np.random.uniform(0, 1, batch_size))
        
        v = np.random.uniform(0, 1, batch_size)
        
        accepted = x_prop[v <= np.sin(np.pi * x_prop)]
        k = len(accepted)
        
        if k > 0:
            limit = min(k, n_samples - accepted_count)
            rejection_samples[accepted_count : accepted_count + limit] = accepted[:limit]
            accepted_count += limit
            
        proposals_count += batch_size
        
    return rejection_samples, n_samples / proposals_count

if __name__ == "__main__":
    bisect_samples = bisection_sampler(N_SAMPLES)
    rejection_samples, acceptance_fraction = rejection_sampler(N_SAMPLES)
    
    print(f"Exact Mean:                     {1 - 4 / (np.pi**2):.3f}")
    print(f"Bisection Sample Mean:          {np.mean(bisect_samples):.3f}")
    print(f"Rejection Sample Mean:          {np.mean(rejection_samples):.3f}")
    print(f"Observed Acceptance Fraction:   {acceptance_fraction:.3f}")