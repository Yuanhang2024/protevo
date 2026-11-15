import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

df = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parent) + r'\data\processed\all_variants.csv')
variants, fitness = df.variant.values, df.fitness.values
AA = 'ACDEFGHIKLMNPQRSTVWY'
X = np.zeros((len(variants), 80))
for i, v in enumerate(variants):
    for p, a in enumerate(v):
        X[i, 20*p + AA.index(a)] = 1

results = []
for seed in [11, 22, 33]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    tr, va, te = idx[:2000], idx[2000:3000], idx[3000:8000]
    # Train on train set (matching v1 protocol)
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features='sqrt', random_state=seed, n_jobs=4)
    rf.fit(X[tr], fitness[tr])
    # Get candidate pool (everything not in train/val)
    pool = idx[3000:]
    # Greedy: score ALL candidates, pick top-10
    preds = rf.predict(X[pool])
    top10_idx = pool[np.argsort(-preds)[:10]]
    top10_fitness = fitness[top10_idx]
    # Random: pick 10 random candidates
    rand10 = pool[rng.choice(len(pool), 10, replace=False)]
    rand10_fitness = fitness[rand10]
    # LLM Agent v1: from original results (mean across 3 rounds)
    results.append({
        'seed': seed,
        'greedy_top10_mean': round(float(np.mean(top10_fitness)), 4),
        'greedy_top10_best': round(float(np.max(top10_fitness)), 4),
        'random_top10_mean': round(float(np.mean(rand10_fitness)), 4),
        'random_top10_best': round(float(np.max(rand10_fitness)), 4,
        ),
    })
    print(f'Seed {seed}: Greedy mean={np.mean(top10_fitness):.3f} best={np.max(top10_fitness):.3f} | Random mean={np.mean(rand10_fitness):.3f}')

df_out = pd.DataFrame(results)
df_out.to_csv(str(__import__("pathlib").Path(__file__).resolve().parent) + r'\results\greedy_baseline.csv', index=False)

# Summary
print('\n=== GREEDY vs LLM Agent (from factorial results) ===')
print(f'Greedy mean:  {np.mean([r["greedy_top10_mean"] for r in results]):.3f}')
print(f'Greedy best:  {np.mean([r["greedy_top10_best"] for r in results]):.3f}')
print(f'LLM v1 mean:  2.084 (from factorial_summary)')
print(f'LLM v3+Tool:  3.840 (from factorial_summary)')
print(f'Random mean:  {np.mean([r["random_top10_mean"] for r in results]):.3f}')
