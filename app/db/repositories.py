"""Parameterized persistence operations for the established SQLite schema."""


class Repositories:
    """Repository facade that preserves existing table and row contracts."""

    def __init__(self, connect):
        self.connect = connect

    def approved_knowledge(self):
        with self.connect() as database:
            return database.execute(
                'SELECT id,title,category,content,source FROM knowledge WHERE approved=1').fetchall()

    def list_knowledge(self):
        with self.connect() as database:
            return database.execute('SELECT * FROM knowledge ORDER BY updated_at DESC,id DESC').fetchall()

    def save_knowledge(self, identifier, title, category, content, source, timestamp):
        with self.connect() as database:
            if identifier is not None:
                cursor = database.execute(
                    'UPDATE knowledge SET title=?,category=?,content=?,source=?,updated_at=? WHERE id=?',
                    (title, category, content, source, timestamp, int(identifier)))
                if cursor.rowcount != 1:
                    raise ValueError('دانش موردنظر یافت نشد')
                return int(identifier)
            return database.execute(
                'INSERT INTO knowledge(title,category,content,source,approved,created_at,updated_at) '
                'VALUES(?,?,?,?,1,?,?)',
                (title, category, content, source, timestamp, timestamp)).lastrowid

    def delete_knowledge(self, identifier):
        with self.connect() as database:
            database.execute('DELETE FROM knowledge WHERE id=?', (int(identifier),))

    def find_document_by_digest(self, digest):
        with self.connect() as database:
            return database.execute('SELECT id FROM documents WHERE sha256=?', (digest,)).fetchone()

    def get_document(self, identifier):
        with self.connect() as database:
            return database.execute(
                'SELECT id,name,sha256,chars,chunks,embed_model,created_at,status,updated_at '
                'FROM documents WHERE id=?', (int(identifier),)).fetchone()

    def create_document_with_chunks(self, name, digest, char_count, chunks, embedding_model, timestamp):
        with self.connect() as database:
            identifier = database.execute(
                'INSERT INTO documents(name,sha256,chars,chunks,embed_model,created_at,status,updated_at) '
                'VALUES (?,?,?,?,?,?,?,?)',
                (name, digest, char_count, len(chunks), embedding_model, timestamp, 'active', timestamp)).lastrowid
            database.executemany(
                '''INSERT INTO doc_chunks(document_id,chunk_no,content,vector,page_start,page_end,heading,
                worksheet,row_start,row_end) VALUES(?,?,?,?,?,?,?,?,?,?)''',
                [self._chunk_insert_values(identifier, chunk) for chunk in chunks])
            return identifier

    def document_records(self):
        with self.connect() as database:
            return database.execute('''SELECT c.document_id,c.chunk_no,c.content,c.vector,c.page_start,c.page_end,
                c.heading,c.worksheet,c.row_start,c.row_end,d.name,d.embed_model
                FROM doc_chunks c JOIN documents d ON d.id=c.document_id''').fetchall()

    def index_records(self):
        """Return only index metadata needed for read-only monitoring statistics."""
        with self.connect() as database:
            return database.execute('''SELECT d.id AS document_id,d.embed_model,
                c.id AS chunk_id,c.vector
                FROM documents d LEFT JOIN doc_chunks c ON c.document_id=d.id''').fetchall()

    def list_documents(self):
        with self.connect() as database:
            return database.execute('''SELECT d.id,d.name,d.sha256,d.chars,d.chunks,d.embed_model,
                d.created_at,d.status,d.updated_at,
                COALESCE(SUM(CASE WHEN c.vector<>'' THEN 1 ELSE 0 END),0) AS vector_chunks
                FROM documents d LEFT JOIN doc_chunks c ON c.document_id=d.id
                GROUP BY d.id ORDER BY d.id DESC''').fetchall()

    def update_document_metadata(self, identifier, name, timestamp):
        with self.connect() as database:
            cursor = database.execute('UPDATE documents SET name=?,updated_at=? WHERE id=?',
                                      (name, timestamp, int(identifier)))
            if cursor.rowcount != 1:
                raise ValueError('سند موردنظر یافت نشد')

    def replace_document_with_chunks(self, identifier, name, digest, char_count, chunks,
                                     embedding_model, timestamp):
        """Replace one selected document atomically while preserving its stable ID."""
        with self.connect() as database:
            target = database.execute('SELECT id FROM documents WHERE id=?', (int(identifier),)).fetchone()
            if not target:
                raise ValueError('سند موردنظر یافت نشد')
            duplicate = database.execute('SELECT id FROM documents WHERE sha256=? AND id<>?',
                                         (digest, int(identifier))).fetchone()
            if duplicate:
                raise ValueError('این فایل قبلاً وارد پایگاه دانش شده است (شناسه ' +
                                 str(duplicate['id']) + ')')
            database.execute('DELETE FROM doc_chunks WHERE document_id=?', (int(identifier),))
            database.execute('''UPDATE documents SET name=?,sha256=?,chars=?,chunks=?,embed_model=?,
                status='active',updated_at=? WHERE id=?''',
                             (name, digest, char_count, len(chunks), embedding_model, timestamp,
                              int(identifier)))
            database.executemany(
                '''INSERT INTO doc_chunks(document_id,chunk_no,content,vector,page_start,page_end,heading,
                worksheet,row_start,row_end) VALUES(?,?,?,?,?,?,?,?,?,?)''',
                [self._chunk_insert_values(int(identifier), chunk) for chunk in chunks])
            return int(identifier)

    @staticmethod
    def _chunk_insert_values(document_id, chunk):
        chunk_no, content, vector = chunk[:3]
        provenance = chunk[3] if len(chunk) > 3 else {}
        return (document_id, chunk_no, content, vector, provenance.get('page_start'),
                provenance.get('page_end'), provenance.get('heading', ''),
                provenance.get('worksheet', ''), provenance.get('row_start'),
                provenance.get('row_end'))

    def delete_document(self, identifier):
        with self.connect() as database:
            cursor = database.execute('DELETE FROM documents WHERE id=?', (int(identifier),))
            if cursor.rowcount != 1:
                raise ValueError('سند موردنظر یافت نشد')

    def list_chats(self):
        with self.connect() as database:
            return database.execute(
                'SELECT id,title,created_at FROM chats ORDER BY created_at DESC').fetchall()

    def chat_history(self, identifier):
        with self.connect() as database:
            return database.execute(
                'SELECT role,content FROM messages WHERE chat_id=? ORDER BY id', (identifier,)).fetchall()

    def recent_messages(self, identifier, limit):
        with self.connect() as database:
            return database.execute(
                'SELECT role,content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?',
                (identifier, limit)).fetchall()

    def chat_exists(self, identifier):
        with self.connect() as database:
            return database.execute('SELECT 1 FROM chats WHERE id=?', (identifier,)).fetchone() is not None

    def create_chat(self, identifier, title, timestamp):
        with self.connect() as database:
            database.execute('INSERT INTO chats(id,title,created_at) VALUES (?,?,?)',
                             (identifier, title, timestamp))

    def store_exchange(self, chat_identifier, user_content, assistant_content, timestamp):
        with self.connect() as database:
            database.execute('INSERT INTO messages(chat_id,role,content,created_at) VALUES(?,?,?,?)',
                             (chat_identifier, 'user', user_content, timestamp))
            database.execute('INSERT INTO messages(chat_id,role,content,created_at) VALUES(?,?,?,?)',
                             (chat_identifier, 'assistant', assistant_content, timestamp))

    def create_chat_with_exchange(self, chat_identifier, title, user_content, assistant_content, timestamp):
        """Persist a successful first turn atomically to avoid empty new chats."""
        with self.connect() as database:
            database.execute('INSERT INTO chats(id,title,created_at) VALUES (?,?,?)',
                             (chat_identifier, title, timestamp))
            database.execute('INSERT INTO messages(chat_id,role,content,created_at) VALUES(?,?,?,?)',
                             (chat_identifier, 'user', user_content, timestamp))
            database.execute('INSERT INTO messages(chat_id,role,content,created_at) VALUES(?,?,?,?)',
                             (chat_identifier, 'assistant', assistant_content, timestamp))
