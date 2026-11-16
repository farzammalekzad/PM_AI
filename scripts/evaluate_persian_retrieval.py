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
from app.services.ollama_client import OllamaClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True, help='Output JSON report path')
    parser.add_argument('--markdown', type=Path, required=True, help='Output Markdown summary path')
    parser.add_argument('--without-embeddings', action='store_true',
                        help='Report semantic and hybrid as unavailable.')
    parser.add_argument('--real-ollama', action='store_true',
                        help='Use only the explicitly named local embedding model.')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--embedding-model', default='')
    arguments = parser.parse_args()
    if arguments.real_ollama:
        client = OllamaClient(arguments.ollama_url)
        model = arguments.embedding_model.strip()
        try:
            if not model or model not in client.list_models():
                raise ValueError('embedding_model_unavailable')
            vector = client.embed('سلام', model)
            embedder = client.embed
            report_kind = 'real_local_embedding'
        except Exception as error:
            report = evaluate_dataset(SYNTHETIC_DOCUMENTS, SYNTHETIC_CASES)
            report['evaluation_kind'] = 'real_local_embedding'
            report['embedding'] = {'model': model, 'available': False,
                                   'dimension': None, 'error': type(error).__name__}
            _write(arguments, report)
            return
        report = evaluate_dataset(SYNTHETIC_DOCUMENTS, SYNTHETIC_CASES, embedder, model)
        report['evaluation_kind'] = report_kind
        report['embedding'] = {'model': model, 'available': True, 'dimension': len(vector), 'error': None}
    else:
        embedder = None if arguments.without_embeddings else synthetic_embedder
        model = '' if arguments.without_embeddings else 'synthetic-persian-evaluation'
        report = evaluate_dataset(SYNTHETIC_DOCUMENTS, SYNTHETIC_CASES, embedder, model)
        report['evaluation_kind'] = 'mocked_embedding' if embedder else 'embedding_unavailable'
    _write(arguments, report)


def _write(arguments, report):
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    arguments.markdown.write_text(markdown_summary(report) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
