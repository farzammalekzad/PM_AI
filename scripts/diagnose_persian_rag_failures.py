"""Create score-level, synthetic-only diagnostics for the two evaluated RAG failures."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.persian_retrieval_dataset import SYNTHETIC_DOCUMENTS, synthetic_embedder
from app.evaluation.retrieval_evaluation import retrieval_diagnostics


QUESTIONS = [
    {'question': 'چه کسی از دیرکرد فعالیت‌ها مطلع می‌شود؟', 'expected_ids': ['D1-C2']},
    {'question': 'رمز عبور سامانه مالی چیست؟', 'expected_ids': []},
]


def _semantic_view(trace):
    candidates = [item for item in trace['candidates'] if item['cosine_similarity'] is not None]
    ranked = [item for item in candidates if item['cosine_similarity'] >= 0.38]
    ranked.sort(key=lambda item: (-item['cosine_similarity'], item['position']))
    return {'top_five': [item['identifier'] for item in ranked[:5]], 'candidates': candidates,
            'threshold_decision': 'include cosine >= 0.38; lexical hits do not affect semantic-only ranking'}


def main():
    output = ROOT / 'reports' / 'task12_retrieval_diagnostics.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    investigations = []
    for item in QUESTIONS:
        lexical = retrieval_diagnostics(SYNTHETIC_DOCUMENTS, item['question'])
        hybrid = retrieval_diagnostics(SYNTHETIC_DOCUMENTS, item['question'], synthetic_embedder,
                                       'synthetic-persian-evaluation')
        investigations.append({
            **item, 'lexical': lexical, 'semantic': _semantic_view(hybrid), 'hybrid': hybrid,
        })
    report = {
        'evaluation_kind': 'mocked_embedding_diagnostics', 'synthetic': True,
        'note': 'Scores are reproducible diagnostics. Mocked embeddings are not real-model results.',
        'investigations': investigations,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
