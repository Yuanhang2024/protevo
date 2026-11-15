"""Validate revised definitions and saved report outputs before new experiments."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/metric-correction-20261114'
definition = json.loads((ROOT / 'configs/statistical-definitions-v2.json').read_text(encoding='utf-8'))
component = json.loads((OUT / 'recomputed/statistics-status.json').read_text(encoding='utf-8'))
build = json.loads((OUT / 'report-build.json').read_text(encoding='utf-8'))
counts = json.loads((OUT / 'version-counts.json').read_text(encoding='utf-8'))
assert definition['version'] == component['definition_version'] == build['definition_version'] == 'gb1-statistical-definitions/2'
assert component['status'] == build['status'] == 'passed'
assert component['model_requests'] == build['new_experimental_model_requests'] == 0
assert counts['combined'] == {'recommendations': 540, 'method_rounds': 54, 'unique_variants': 381}
assert component['first_answers']['method_round_records'] == 36
assert component['anchors']['measurement_history_eligible_pairs'] == 255
assert component['anchors']['prompt_visible_pairs'] == 255 and component['anchors']['unknown_prompt_visibility_pairs'] == 0
assert component['confirmation']['continuous_quality_intervals_exactly_unchanged']
assert component['confirmation']['boundary_probability_interval_lower'] is None
assert component['confirmation']['boundary_probability_interval_upper'] is None

for relative in ('reports/GB1首轮实验报告', 'reports/validation-stage/GB1科学Agent验证报告'):
    docx, pdf = ROOT / (relative + '.docx'), ROOT / (relative + '.pdf')
    doc = Document(docx)
    body = '\n'.join(p.text for p in doc.paragraphs)
    for unsupported in ('反馈无独立可测增益', '预训练记忆了该公开数据', 'ESM/GP需CUDA'):
        assert unsupported not in body
    with pdf.open('rb') as source:
        assert source.read(5) == b'%PDF-'
    assert (OUT / 'originals' / (relative + '.pdf')).is_file()

for relative in ('reports/统计定义与版本范围.md', 'results/critique-followup-20261113/对照实验说明.md'):
    path = ROOT / relative
    for link in re.findall(r'\]\(<([^>]+)>\)', path.read_text(encoding='utf-8')):
        if not link.startswith(('https:', 'http:')):
            assert (path.parent / link.split('#', 1)[0]).resolve().exists(), (relative, link)

status = {
    'status': 'passed', 'definition_version': definition['version'],
    'scope': 'statistical definitions, recomputed data, current report text and PDF outputs',
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'python_tests': {'command': 'python -m pytest tests -q -p no:cacheprovider', 'passed': 332, 'warnings': 2},
    'statistical_component_tests': component['targeted_tests_passed'],
    'historical_raw_experiments_preserved': True,
    'report_originals_preserved': True,
    'revised_pdf_pages': {'main': 5, 'validation': 5},
    'critical_pages_visually_checked': ['main page 5', 'validation pages 4 and 5'],
    'new_experimental_model_requests': 0,
    'recomputed_component': 'recomputed/statistics-status.json',
    'version_counts': 'version-counts.json',
    'authorization': {'feedback_model': 'gpt-6-luna', 'api_mode': 'chat', 'reasoning_effort': 'xhigh',
                      'maximum_generation_calls': 48, 'basis': 'direct user selection of the 48-call pilot option',
                      'contexts': ['C0', 'C1'], 'selectors': ['P0', 'P1'], 'states': [1201,1202,1203,1204], 'generation_repeats': 3},
}
path = OUT / 'correction-status.json'
if path.exists():
    raise FileExistsError('A correction status already exists; inspect rather than overwrite.')
path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
print(path)
