"""Chat orchestration that preserves the existing NDJSON event contract."""
import re
import uuid
from dataclasses import dataclass
from datetime import datetime

from app.services.citation_service import source_metadata


@dataclass
class ChatSession:
    """An opened local generation whose events can be forwarded to a client."""

    chat_id: str
    is_new_chat: bool
    title: str
    content: str
    facts: list
    documents: list
    upstream: object
    repository: object
    diagnostics: object

    def events(self):
        """Yield source and content events, persisting only non-empty completed output."""
        yield {
            'chat_id': self.chat_id,
            **source_metadata(self.facts, self.documents),
        }
        full = []
        completed = False
        for chunk in self.upstream:
            if chunk.get('error'):
                yield {'error': str(chunk['error'])}
                return
            message = chunk.get('message') or {}
            if not isinstance(message, dict):
                raise ValueError('پاسخ پخش جریانی Ollama نامعتبر است')
            fragment = message.get('content', '')
            if not isinstance(fragment, str):
                raise ValueError('پاسخ پخش جریانی Ollama نامعتبر است')
            if fragment:
                full.append(fragment)
            yield {
                'content': fragment,
                'thinking': message.get('thinking', ''),
                'done': bool(chunk.get('done')),
            }
            if chunk.get('done'):
                completed = True
        answer = ''.join(full)
        if completed and answer:
            timestamp = datetime.now().isoformat(timespec='seconds')
            if self.is_new_chat:
                self.repository.create_chat_with_exchange(
                    self.chat_id, self.title, self.content, answer, timestamp)
            else:
                self.repository.store_exchange(self.chat_id, self.content, answer, timestamp)


class ChatService:
    """Coordinate retrieval, prompt construction, local streaming, and persistence."""

    def __init__(self, repository_factory, ollama_client_factory, search_knowledge,
                 search_documents, prompt_builder, allowed_context_windows):
        self.repository_factory = repository_factory
        self.ollama_client_factory = ollama_client_factory
        self.search_knowledge = search_knowledge
        self.search_documents = search_documents
        self.prompt_builder = prompt_builder
        self.allowed_context_windows = tuple(allowed_context_windows)

    def start(self, request):
        """Validate and open a local stream before any new chat row is created."""
        model = str(request.get('model', '')).strip()
        chat_id = request.get('chat_id')
        if chat_id is not None and not re.fullmatch(r'[a-f0-9]{32}', str(chat_id)):
            raise ValueError('شناسه گفت‌وگو نامعتبر است')
        question = str(request.get('question', '')).strip()
        content = str(request.get('content', '')).strip()
        context_window = int(request.get('ctx', 8192))
        if (not model or not question or not content or
                context_window not in self.allowed_context_windows or len(content) > 120000):
            raise ValueError('پارامترهای مدل یا پرسش نامعتبرند')

        facts = self.search_knowledge(question)
        documents = self.search_documents(question, limit=5)
        repository = self.repository_factory()
        is_new_chat = chat_id is None
        if is_new_chat:
            chat_id = uuid.uuid4().hex
            history = []
        else:
            if not repository.chat_exists(chat_id):
                raise ValueError('گفت‌وگو یافت نشد')
            rows = repository.recent_messages(chat_id, 16)
            history = [{'role': row['role'], 'content': row['content']} for row in reversed(rows)]

        prompt = self.prompt_builder.build(content, facts, documents, history, context_window)
        upstream = self.ollama_client_factory().stream_chat(model, prompt.messages, context_window)
        return ChatSession(
            chat_id=chat_id,
            is_new_chat=is_new_chat,
            title=question[:75],
            content=content,
            facts=facts,
            documents=documents,
            upstream=upstream,
            repository=repository,
            diagnostics=prompt.diagnostics,
        )
