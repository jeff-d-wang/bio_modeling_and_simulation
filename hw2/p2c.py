import math

OBS_RELEVANT = 18
OBS_IRRELEVANT = 2
TOTAL_OBS = 500

def get_norm_posterior():
    unnorm_post = {}
    for k in range(TOTAL_OBS + 1):
        if k < OBS_RELEVANT or k > TOTAL_OBS - OBS_IRRELEVANT:
            unnorm_post[k] = 0
        else:
            unnorm_post[k] = math.comb(k, OBS_RELEVANT) * math.comb(TOTAL_OBS - k, OBS_IRRELEVANT)
    
    total_sum = sum(unnorm_post.values())
    post = {k: v / total_sum for k, v in unnorm_post.items()}
    
    post_mode = max(post, key=post.get)
    
    post_mean = sum(k * p for k, p in post.items())
    
    # 90% CI
    cum_prob = 0
    k_lo = None
    k_hi = None
    for k in range(TOTAL_OBS + 1):
        cum_prob += post[k]
        if k_lo is None and cum_prob >= 0.05:
            k_lo = k
        if k_hi is None and cum_prob >= 0.95:
            k_hi = k
            
    return post_mode, post_mean, k_lo, k_hi

post_mode, post_mean, k_lo, k_hi = get_norm_posterior()
print(f"Posterior mode: {post_mode}")
print(f"Posterior mean: {post_mean}")
print(f"90% CI: [{k_lo}, {k_hi}]")