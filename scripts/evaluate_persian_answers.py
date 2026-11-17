"""Evaluate synthetic grounded Persian answers; local Gemma use is explicit and opt-in."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.answer_evaluation import evaluate_answers
from app.evaluation.persian_answer_dataset import ANSWER_CASES, MOCKED_ANSWERS
from app.evaluation.persian_retrieval_dataset import SYNTHETIC_DOCUMENTS
from app.services.citation_service import document_identifier
from app.services.ollama_client import OllamaClient
from app.services.prompt_builder import PromptBuilder


def _documents(identifiers):
    wanted = set(identifiers)
    return [item for item in SYNTHETIC_DOCUMENTS if document_identifier(item) in wanted]


def _local_answers(url, model):
    client = OllamaClient(url)
    if not model or model not in client.list_models():
        raise ValueError('chat_model_unavailable')
    answers = {}
    for case in ANSWER_CASES:
        prompt = PromptBuilder().build(case['question'], [], _documents(case['retrieved_ids']), [], 8192)
        parts = []
        for event in client.stream_chat(model, prompt.messages, 8192):
            message = event.get('message') or {}
            content = message.get('content', '') if isinstance(message, dict) else ''
            if isinstance(content, str):
                parts.append(content)
        answers[case['id']] = ''.join(parts)
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--real-ollama', action='store_true')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--chat-model', default='')
    arguments = parser.parse_args()
    if arguments.real_ollama:
        try:
            answers = _local_answers(arguments.ollama_url, arguments.chat_model)
        except Exception as error:
            report = evaluate_answers(ANSWER_CASES, {}, 'real_local_gemma_unavailable')
            report['chat'] = {'model': arguments.chat_model, 'available': False,
                              'error': type(error).__name__}
        else:
            # _local_answers returns answers; keep this branch explicit to avoid mock substitution.
            report = evaluate_answers(ANSWER_CASES, answers, 'real_local_gemma')
            report['chat'] = {'model': arguments.chat_model, 'available': True, 'error': None}
    else:
        report = evaluate_answers(ANSWER_CASES, MOCKED_ANSWERS, 'mocked_answers')
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
