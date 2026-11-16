"""Run the reproducible synthetic Persian retrieval evaluation offline."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.persian_retrieval_dataset import (
    SYNTHETIC_CASES, SYNTHETIC_DOCUMENTS, synthetic_embedder,
)
from app.evaluation.retrieval_evaluation import evaluate_dataset, markdown_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True, help='Output JSON report path')
    parser.add_argument('--markdown', type=Path, required=True, help='Output Markdown summary path')
    parser.add_argument('--without-embeddings', action='store_true',
                        help='Report semantic and hybrid as unavailable.')
    arguments = parser.parse_args()
    embedder = None if arguments.without_embeddings else synthetic_embedder
    model = '' if arguments.without_embeddings else 'synthetic-persian-evaluation'
    report = evaluate_dataset(SYNTHETIC_DOCUMENTS, SYNTHETIC_CASES, embedder, model)
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    arguments.markdown.write_text(markdown_summary(report) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
