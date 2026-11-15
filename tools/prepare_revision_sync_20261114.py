"""Prepare safe local release synchronization without Git operations."""
from pathlib import Path
import json
import shutil

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT.parent / 'gb1-scientific-agent-release'
ARCHIVE = PROJECT / 'results/metric-correction-20261114/originals'
OUT = PROJECT / 'results/metric-correction-20261114/release-sync'
OLD = OUT / 'before'
EXISTING = [
    'build_report.py', 'build_validation_report.py',
    'results/critique-followup-20261113/zero_context.py',
    'results/critique-followup-20261113/analyze_stats.py',
    'results/critique-followup-20261113/no_feedback.py',
    'results/critique-followup-20261113/对照实验说明.md',
    'reports/validation-stage/验证结果.md', 'reports/实验结果.md',
    'reports/GB1首轮实验报告.pdf', 'reports/validation-stage/GB1科学Agent验证报告.pdf',
]
NEW = [
    'protevo/statistical_definitions.py', 'configs/statistical-definitions-v2.json',
    'reports/统计定义与版本范围.md',
    'tools/reassessment_20261114/recompute_saved_statistics.py',
    'tools/supplement_20261114/strong_baselines.py', 'tools/supplement_20261114/feedback_pilot.py',
    'tests/test_statistical_definitions.py', 'tests/test_saved_ranking_statistics.py',
    'tests/test_strong_baselines_20261114.py', 'tests/test_feedback_pilot_20261114.py',
    'tools/calibration-report/audit.py', 'tools/calibration-report/plots.py',
    'tools/calibration-report/report.py', 'tools/calibration-report/statistics_report.py',
    'tools/calibration-report/test_report.py',
    'results/metric-correction-20261114/recomputed/statistics-status.json',
    'results/metric-correction-20261114/recomputed/zero-context/summary.json',
    'results/metric-correction-20261114/recomputed/zero-context/ranking.csv',
    'results/metric-correction-20261114/recomputed/first-answer-anchors/stats_summary.json',
    'results/metric-correction-20261114/recomputed/first-answer-anchors/first_answer_overlap.csv',
    'results/metric-correction-20261114/recomputed/first-answer-anchors/anchor_transfer.csv',
    'results/metric-correction-20261114/recomputed/first-answer-anchors/anchor_transfer_prompt_visible.csv',
    'results/metric-correction-20261114/recomputed/completion/statistics.json',
    'results/metric-correction-20261114/version-counts.json',
]


def same(a, b):
    if a.suffix == '.pdf':
        return a.read_bytes() == b.read_bytes()
    return a.read_text(encoding='utf-8-sig').replace('\r\n', '\n') == b.read_text(encoding='utf-8-sig').replace('\r\n', '\n')


def main():
    if not (RELEASE / '.git').is_dir():
        raise ValueError('The expected release repository is absent')
    if OUT.exists():
        raise FileExistsError('Inspect the existing synchronization record rather than overwriting it')
    OUT.mkdir(parents=True)
    rows = []
    for relative in EXISTING + NEW:
        source, target = PROJECT / relative, RELEASE / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        if not target.resolve().is_relative_to(RELEASE.resolve()):
            raise ValueError('Unexpected release path')
        if target.exists() and same(source, target):
            rows.append({'path': relative, 'action': 'already_current'})
            continue
        if target.exists():
            previous = ARCHIVE / relative
            if relative not in EXISTING or not previous.is_file() or not same(target, previous):
                rows.append({'path': relative, 'action': 'preserved_independent_release_change'})
                continue
            saved = OLD / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, saved)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({'path': relative, 'action': 'updated' if relative in EXISTING else 'added'})
    rows.append({'path': 'README.md', 'action': 'independent_release_readme_preserved_for_targeted_edit'})
    (OUT / 'sync-actions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'updated_or_added': sum(r['action'] in ('updated', 'added') for r in rows),
                      'independent_changes_preserved': [r['path'] for r in rows if r['action'].startswith('preserved_')],
                      'git_actions': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
