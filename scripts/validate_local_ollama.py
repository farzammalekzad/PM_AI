"""Opt-in, non-destructive validation of locally installed Ollama models."""
import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ollama_client import OllamaClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ollama-url', default=os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434'))
    parser.add_argument('--embedding-model', default=os.environ.get('OLLAMA_EMBED_MODEL', ''))
    parser.add_argument('--chat-model', default='', help='Optional installed Gemma model name.')
    arguments = parser.parse_args()
    client = OllamaClient(arguments.ollama_url, tags_timeout=5, embedding_timeout=30, chat_timeout=60)
    report = {'url': arguments.ollama_url, 'models': [], 'embedding': None, 'chat': None}
    try:
        models = client.list_models()
        report['models'] = models
    except Exception as error:
        report['connection_error'] = type(error).__name__
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    model = arguments.embedding_model.strip()
    if not model:
        report['embedding'] = {'status': 'not_configured'}
    elif model not in models:
        report['embedding'] = {'status': 'model_not_found', 'model': model}
    else:
        try:
            report['embedding'] = {'status': 'available', 'model': model,
                                   'dimension': len(client.embed('سلام', model))}
        except Exception as error:
            report['embedding'] = {'status': 'unavailable', 'model': model,
                                   'error': type(error).__name__}
    chat_model = arguments.chat_model.strip() or next((name for name in models if 'gemma' in name.lower()), '')
    if not chat_model:
        report['chat'] = {'status': 'gemma_model_not_found'}
    else:
        try:
            chunks = list(client.stream_chat(chat_model, [{'role': 'user', 'content': 'سلام، پاسخ کوتاه بده.'}], 4096))
            report['chat'] = {'status': 'available', 'model': chat_model,
                              'received_chunks': len(chunks)}
        except Exception as error:
            report['chat'] = {'status': 'unavailable', 'model': chat_model,
                              'error': type(error).__name__}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
