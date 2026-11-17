"""Measure the opt-in, synthetic Persian synonym experiment without production changes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.persian_retrieval_dataset import SYNTHETIC_CASES, SYNTHETIC_DOCUMENTS
from app.evaluation.retrieval_evaluation import evaluate_synonym_experiment


def main():
    output = ROOT / 'reports' / 'task13_synonym_experiment.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate_synonym_experiment(SYNTHETIC_DOCUMENTS, SYNTHETIC_CASES)
    report['note'] = ('Opt-in synthetic experiment only; it does not alter production ranking or thresholds.')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
