"""Copy audited supplement summaries and public records into the release copy."""
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT.parent / 'gb1-scientific-agent-release'
FILES = [
    'results/supplement-20261114/补充实验预先方案.md',
    'results/supplement-20261114/全池模型对照结果.md',
    'results/supplement-20261114/反馈预试结果.md',
    'results/supplement-20261114/public-records-audit.json',
    'results/supplement-20261114/strong-baselines/strategy_summary.csv',
    'results/supplement-20261114/strong-baselines/per_seed_strategy_metrics.csv',
    'results/supplement-20261114/strong-baselines/audit.json',
    'results/supplement-20261114/feedback-pilot/summary.json',
    'results/supplement-20261114/feedback-pilot/arm-audit.csv',
]
PUBLIC = PROJECT / 'results/supplement-20261114/public-records'

for relative in FILES:
    source, target = PROJECT / relative, RELEASE / relative
    if not source.is_file():
        raise FileNotFoundError(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

target_root = RELEASE / 'results/supplement-20261114/public-records'
if target_root.exists() and any(target_root.iterdir()):
    raise FileExistsError('Public records already exported; inspect instead of overwriting')
count = 0
for source in PUBLIC.rglob('*'):
    if source.is_file():
        target = target_root / source.relative_to(PUBLIC)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        count += 1
print(f'Copied {len(FILES)} summaries and {count} public-record files into the release copy.')
