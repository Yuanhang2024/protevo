import numpy as np
from scipy.stats import studentized_range

# Nemenyi: q_obs = |mean rank diff| / sqrt(k(k+1)/(6N)); two-sided p = 1 - CDF
k, N = 4, 9
se = np.sqrt(k * (k + 1) / (6 * N))
print("SE =", round(se, 6))
# published: CD_alpha0.05 = 1.563463 -> implied q_crit
print("implied q_crit from CD 1.563463:", round(1.563463 / se, 4))
print("studentized_range.cdf(2.5691, 4, inf) =", round(studentized_range.cdf(1.563463 / se, 4, np.inf), 6))
for diff, pub in [(1.9444, 0.007635), (1.8333, 0.013806), (1.9444 - 0, 0.05175)]:
    q = diff / se
    p = 2 * (1 - studentized_range.cdf(q, k, np.inf))
    print(f"|diff|={diff}: q={q:.4f} two-sided p={p:.6f} (published {pub})")
# random-knowledge mean_fitness diff: ranks 1.3333 vs 2.8889 = 1.5556, published 0.05175
for diff, pub in [(1.5556, 0.05175), (2.2222, 0.001487)]:
    q = diff / se
    p = 2 * (1 - studentized_range.cdf(q, k, np.inf))
    print(f"|diff|={diff}: q={q:.4f} two-sided p={p:.6f} (published {pub})")
