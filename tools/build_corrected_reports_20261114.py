"""Regenerate current report text from archived, completed experiment records."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from docx import Document
import build_report as main_report
import build_validation_report as validation_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tests-passed', type=int, required=True)
args = parser.parse_args()
if args.tests_passed < 1:
    raise ValueError('Use the actual successful test-run count')

summary, recommendations, rounds, final, usage, states = main_report.collect()
assert len(recommendations) == 360 and len(rounds) == 36
main_docx = main_report.make_docx(summary, recommendations, rounds, final, usage, states)
main_report.make_markdown(summary, recommendations, usage)

result, records, pairs, arms = validation_report.prepare_data()
assert len(records) == 180 and len(pairs) == 3 and len(arms) == 6
validation_docx = validation_report.document(result, records, pairs, arms, args.tests_passed)

for path in (main_docx, validation_docx):
    doc = Document(path)
    body = '\n'.join(p.text for p in doc.paragraphs)
    assert '反馈无独立可测增益' not in body
    assert '预训练记忆了该公开数据' not in body
    assert 'ESM/GP需CUDA' not in body
    assert '统计定义v2' in '\n'.join(p.text for section in doc.sections for p in section.header.paragraphs)

output = ROOT / 'results/metric-correction-20261114/report-build.json'
output.write_text(json.dumps({'status': 'passed', 'definition_version': 'gb1-statistical-definitions/2',
    'python_tests': {'command': 'python -m pytest tests -q -p no:cacheprovider', 'passed': args.tests_passed},
    'v1_recommendations': len(recommendations), 'v1_method_rounds': len(rounds),
    'v3_recommendations': len(records), 'detailed_feedback_states': len(pairs),
    'documents': [str(Path(main_docx).relative_to(ROOT)), str(Path(validation_docx).relative_to(ROOT))],
    'new_experimental_model_requests': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
print(output)
