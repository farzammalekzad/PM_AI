"""Run direct, oracle, and RAG grounded-answer evaluation against explicit local Gemma."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.gemma_grounding_evaluation import run_gemma_experiments, unavailable_report
from app.evaluation.persian_answer_dataset import ANSWER_CASES
from app.evaluation.persian_retrieval_dataset import SYNTHETIC_DOCUMENTS
from app.services.ollama_client import OllamaClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--chat-model', default='gemma4:e4b-it-qat')
    parser.add_argument('--embedding-model', default='qwen3-embedding-local:latest')
    arguments = parser.parse_args()
    try:
        client = OllamaClient(arguments.ollama_url)
        models = client.list_models()
        if arguments.chat_model not in models:
            raise ValueError('chat_model_unavailable')
        if arguments.embedding_model not in models:
            raise ValueError('embedding_model_unavailable')
        report = run_gemma_experiments(client, arguments.chat_model, ANSWER_CASES,
                                       SYNTHETIC_DOCUMENTS, embedding_model=arguments.embedding_model)
    except Exception as error:
        report = unavailable_report(arguments.chat_model, type(error).__name__)
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
