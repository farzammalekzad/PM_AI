"""Small standard-library client for the local Ollama HTTP API."""
import json
import math
import urllib.error
import urllib.request


class OllamaResponseError(ValueError):
    """Raised when Ollama returns an invalid HTTP or JSON response."""


class OllamaClient:
    """Encapsulate local Ollama calls without model selection side effects."""

    def __init__(self, base_url, tags_timeout=8, embedding_timeout=240, chat_timeout=1200):
        self.base_url = base_url.rstrip('/')
        self.tags_timeout = tags_timeout
        self.embedding_timeout = embedding_timeout
        self.chat_timeout = chat_timeout

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        return urllib.request.Request(
            self.base_url + path, data=data,
            headers={'Content-Type': 'application/json'} if data is not None else {},
            method='POST' if data is not None else 'GET')

    def list_models(self):
        try:
            with urllib.request.urlopen(self._request('/api/tags'), timeout=self.tags_timeout) as response:
                payload = json.load(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaResponseError('پاسخ Ollama نامعتبر است') from error
        if not isinstance(payload, dict):
            raise OllamaResponseError('پاسخ Ollama نامعتبر است')
        models = payload.get('models', [])
        if not isinstance(models, list):
            raise OllamaResponseError('پاسخ Ollama نامعتبر است')
        return [item['name'] for item in models if isinstance(item, dict) and isinstance(item.get('name'), str)]

    def is_available(self):
        try:
            self.list_models()
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return False

    def embed(self, text, model):
        request = self._request('/api/embed', {'model': model, 'input': text})
        try:
            with urllib.request.urlopen(request, timeout=self.embedding_timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(800).decode('utf-8', 'replace')
            raise OllamaResponseError('خطای مدل Embedding: ' + detail) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaResponseError('Embedding معتبر دریافت نشد') from error
        embeddings = payload.get('embeddings') if isinstance(payload, dict) else None
        vector = embeddings[0] if isinstance(embeddings, list) and embeddings else None
        if not isinstance(vector, list) or not vector:
            raise OllamaResponseError('Embedding معتبر دریافت نشد')
        try:
            normalized = [float(value) for value in vector]
            magnitude = math.sqrt(sum(value * value for value in normalized))
        except (TypeError, ValueError) as error:
            raise OllamaResponseError('Embedding معتبر دریافت نشد') from error
        if not magnitude:
            raise OllamaResponseError('Embedding صفر است')
        return [round(value / magnitude, 7) for value in normalized]

    def stream_chat(self, model, messages, context_window):
        """Yield validated NDJSON objects; callers decide how to persist output."""
        request = self._request('/api/chat', {
            'model': model,
            'stream': True,
            'messages': messages,
            'options': {'num_ctx': context_window},
        })
        try:
            response = urllib.request.urlopen(request, timeout=self.chat_timeout)
        except urllib.error.HTTPError as error:
            detail = error.read(1500).decode('utf-8', 'replace')
            raise OllamaResponseError('خطای Ollama: ' + detail) from error
        def chunks():
            with response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        payload = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        raise OllamaResponseError('پاسخ پخش جریانی Ollama نامعتبر است') from error
                    if not isinstance(payload, dict):
                        raise OllamaResponseError('پاسخ پخش جریانی Ollama نامعتبر است')
                    yield payload
        return chunks()
