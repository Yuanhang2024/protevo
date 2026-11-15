"""Build report figures from saved predictor and evolutionary experiment records."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'figures-v2-data'
SEEDS = (11, 22, 33)
GROUPS = ('G1', 'G2', 'G3', 'G4')
COLORS = ('#306998', '#d17a22', '#538659', '#925b91')
RULE = 'Within 50000 training examples, select the model with highest mean validation Spearman across seeds 11, 22, 33; ties resolved by model name.'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def predictor_figures():
    data = pd.read_csv(ROOT / 'results/predictor_v2/full_ablation.csv')
    keys = ['feature', 'model', 'train_size', 'seed']
    assert not data.duplicated(keys).any()
    assert np.isfinite(data.val_spearman).all()
    d = data[data.train_size == 50000].copy()
    assert all(set(g.seed) == set(SEEDS) and len(g) == len(SEEDS) for _, g in d.groupby(['feature', 'model']))
    stats = d.groupby(['feature', 'model'], as_index=False).agg(mean=('val_spearman', 'mean'), sd=('val_spearman', 'std'), n=('seed', 'size'))
    chosen = stats.sort_values(['feature', 'mean', 'model'], ascending=[True, False, True]).drop_duplicates('feature').sort_values('mean', ascending=True)
    chosen['train_size'] = 50000
    chosen['selection_rule'] = RULE
    chosen.to_csv(OUT / 'predictor_50k_selection.csv', index=False)
    points = d.merge(chosen[['feature', 'model']], on=['feature', 'model'], validate='many_to_one').sort_values(['feature', 'seed'])
    points.to_csv(OUT / 'predictor_50k_seed_values.csv', index=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for i, row in enumerate(chosen.itertuples()):
        vals = points.loc[(points.feature == row.feature) & (points.model == row.model), 'val_spearman']
        ax.scatter(vals, np.full(len(vals), i), color='#306998', s=27, alpha=.8, zorder=3)
        ax.errorbar(row.mean, i, xerr=row.sd, fmt='D', color='#b44e36', capsize=3, markersize=4, zorder=4)
    ax.set_yticks(range(len(chosen)), [f'{r.feature} / {r.model}' for r in chosen.itertuples()])
    ax.set_xlabel('Validation Spearman correlation')
    ax.set_title('50,000 training examples: model selected within each feature\nSeed values and mean ± sample SD (n = 3)', loc='left', fontsize=10)
    ax.grid(axis='x', alpha=.2)
    save(fig, 'predictor_50k')
    pairs = [('onehot_80d', 'rf'), ('L33_mutpos', 'mlp'), ('8m_mutpos_320', 'mlp')]
    curve = pd.concat([data[(data.feature == f) & (data.model == m)] for f, m in pairs])
    assert set(curve.train_size) == {2000, 10000, 50000}
    assert len(curve) == 27
    assert all(set(g.seed) == set(SEEDS) and len(g) == 3 for _, g in curve.groupby(['feature', 'model', 'train_size']))
    curve.to_csv(OUT / 'predictor_training_size_seed_values.csv', index=False)
    agg = curve.groupby(['feature', 'model', 'train_size'], as_index=False).agg(mean=('val_spearman', 'mean'), sd=('val_spearman', 'std'), n=('seed', 'size'))
    agg.to_csv(OUT / 'predictor_training_size_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for (feature, model), color in zip(pairs, COLORS):
        z = agg[(agg.feature == feature) & (agg.model == model)].sort_values('train_size')
        ax.errorbar(z.train_size, z['mean'], yerr=z.sd, label=f'{feature} / {model}', color=color, marker='o', capsize=3)
    ax.set_xscale('log')
    ax.set_xticks([2000, 10000, 50000], ['2,000', '10,000', '50,000'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel='Number of training examples', ylabel='Validation Spearman correlation', title='Fixed feature and model pairs: mean ± sample SD (n = 3)')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, 'predictor_training_size')


def state_path(group, seed):
    if group == 'G1':
        return ROOT / f'results/seed-{seed}/run/state.json'
    if group == 'G2':
        return ROOT / f'results/v2_experiments/G2/seed-{seed}/run/state.json'
    return ROOT / f'results/v2_experiments/{group}/seed-{seed}/state.json'


def factorial_and_examples():
    saved = pd.read_csv(ROOT / 'results/v2_experiments/factorial_summary.csv').query("level == 'seed'")
    rows, matches = [], []
    for group in GROUPS:
        for seed in SEEDS:
            state = json.loads(state_path(group, seed).read_text(encoding='utf-8'))
            rounds = state['methods']['llm']['rounds'] if group in ('G1', 'G2') else state['rounds']
            assert len(rounds) == 3
            recs = [q for r in rounds for q in r['recommendations']]
            assert len(recs) == 30
            vals = np.array([q['actual_fitness'] for q in recs], dtype=float)
            train = pd.read_csv(ROOT / f'data/processed/seed-{seed}/train.csv')
            mean = vals.mean()
            best = max(train.fitness.max(), vals.max())
            summary = saved[(saved.group == group) & (saved.seed == seed)]
            assert len(summary) == 1
            assert np.isclose(mean, summary.iloc[0].mean_round_fitness, atol=1e-9)
            assert np.isclose(best, summary.iloc[0].final_cumulative_best, atol=1e-9)
            rows.append({'group': group, 'seed': seed, 'trajectory_mean': mean, 'final_cumulative_best': best, 'recommendations': len(recs)})
            if group not in ('G1', 'G2'):
                continue
            for r in rounds:
                proposal_ids = {p['variant'] for p in r['proposals']}
                origins, examples = {}, set()
                for attempt in r['attempts']:
                    try:
                        body = json.loads(attempt['response']['text'])
                    except (KeyError, ValueError, TypeError):
                        body = {}
                    raw = body.get('proposals', []) if isinstance(body, dict) else []
                    added = []
                    if attempt.get('valid_count', 0):
                        for q in raw:
                            variant = q.get('variant') if isinstance(q, dict) else None
                            if variant in proposal_ids and variant not in origins:
                                origins[variant] = attempt['number']
                                added.append(variant)
                    assert len(added) == attempt.get('valid_count', 0)
                    context = attempt['context']
                    shown = set(context.get('candidate_examples', [])) | set(context.get('replacement_examples', []))
                    examples.update(set(added) & shown)
                assert set(origins) == proposal_ids
                selected = {p['variant'] for p in r['recommendations']}
                assert selected <= proposal_ids
                matches.extend([{'group': group, 'seed': seed, 'round': r['round'], 'population': 'Accepted candidates', 'matched': len(examples), 'total': len(proposal_ids)}, {'group': group, 'seed': seed, 'round': r['round'], 'population': 'Final recommendations', 'matched': len(selected & examples), 'total': len(selected)}])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'factorial_seed_values.csv', index=False)
    summaries = frame.groupby('group', as_index=False).agg(trajectory_mean=('trajectory_mean', 'mean'), trajectory_sd=('trajectory_mean', 'std'), cumulative_best_mean=('final_cumulative_best', 'mean'), cumulative_best_sd=('final_cumulative_best', 'std'), n=('seed', 'size'))
    summaries.to_csv(OUT / 'factorial_summary.csv', index=False)
    for metric, title, ylabel in [('trajectory_mean', 'Mean fitness of 30 recommendations', 'Mean measured fitness'), ('final_cumulative_best', 'Final cumulative best including initial training set', 'Best measured fitness')]:
        fig, ax = plt.subplots(figsize=(6.3, 4.1))
        for i, group in enumerate(GROUPS):
            v = frame[frame.group == group][metric].to_numpy()
            ax.scatter(np.full(len(v), i) + np.linspace(-.09, .09, len(v)), v, s=38, color=COLORS[i], alpha=.75)
            ax.errorbar(i, np.mean(v), yerr=np.std(v, ddof=1), color='#222222', marker='D', capsize=4, markersize=4)
        ax.set_xticks(range(4), GROUPS)
        ax.set(xlabel='Experiment group', ylabel=ylabel, title=title + '\nSeed values and mean ± sample SD (n = 3)')
        ax.grid(axis='y', alpha=.2)
        save(fig, 'factorial_' + metric)
    details = pd.DataFrame(matches)
    details.to_csv(OUT / 'example_match_round_values.csv', index=False)
    totals = details.groupby(['group', 'population'], as_index=False)[['matched', 'total']].sum()
    totals['fraction'] = totals.matched / totals.total
    assert {(r.group, r.population): (r.matched, r.total) for r in totals.itertuples()} == {('G1', 'Accepted candidates'): (200, 358), ('G1', 'Final recommendations'): (48, 90), ('G2', 'Accepted candidates'): (325, 334), ('G2', 'Final recommendations'): (81, 90)}
    totals.to_csv(OUT / 'example_match_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.7, 4.1))
    for i, pop in enumerate(('Accepted candidates', 'Final recommendations')):
        for j, group in enumerate(('G1', 'G2')):
            row = totals[(totals.population == pop) & (totals.group == group)].iloc[0]
            x = i + (j - .5) * .29
            ax.bar(x, row.fraction * 100, width=.25, color=COLORS[j], label=group if i == 0 else None)
            ax.text(x, row.fraction * 100 + 2, f'{row.matched}/{row.total}', ha='center', fontsize=9)
    ax.set_xticks([0, 1], ['Accepted candidates', 'Final recommendations'])
    ax.set_ylim(0, 110)
    ax.set(ylabel='Matching shown examples (%)', title='G1 and G2: matches to examples shown in originating attempt')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2)
    save(fig, 'example_match')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    predictor_figures()
    factorial_and_examples()
    tables = ('predictor_50k_selection', 'predictor_training_size_summary', 'factorial_summary', 'example_match_summary')
    metrics = {name: pd.read_csv(OUT / f'{name}.csv').to_dict(orient='records') for name in tables}
    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Figures and data saved to {OUT}')


if __name__ == '__main__':
    main()
