"""Preserve selected current source and report files before statistical revision."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/metric-correction-20261114'
ARCHIVE = OUT / 'originals'
FILES = [
    'README.md', 'build_report.py', 'build_validation_report.py',
    'results/critique-followup-20261113/zero_context.py',
    'results/critique-followup-20261113/analyze_stats.py',
    'results/critique-followup-20261113/no_feedback.py',
    'results/critique-followup-20261113/对照实验说明.md',
    'tools/calibration-report/statistics_report.py',
    'reports/实验结果.md', 'reports/GB1首轮实验报告.pdf', 'reports/GB1首轮实验报告.docx',
    'reports/validation-stage/验证结果.md',
    'reports/validation-stage/GB1科学Agent验证报告.pdf',
    'reports/validation-stage/GB1科学Agent验证报告.docx',
]
if ARCHIVE.exists():
    raise FileExistsError('Revision archive already exists; inspect it instead of overwriting.')
ARCHIVE.mkdir(parents=True, exist_ok=False)
records = []
for relative in FILES:
    source = ROOT / relative
    if not source.is_file():
        records.append({'path': relative, 'present': False})
        continue
    destination = ARCHIVE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    records.append({'path': relative, 'present': True, 'bytes': source.stat().st_size})
(OUT / 'preserved-files.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'preserved_files': sum(row['present'] for row in records), 'directory': str(ARCHIVE)}, ensure_ascii=False))
