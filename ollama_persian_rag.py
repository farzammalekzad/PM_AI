# -*- coding: utf-8 -*-
"""Offline Persian Ollama chat with SQLite knowledge and conversation history.
Python 3.10+, standard library only. Optional pypdf for text PDFs.
Run: python ollama_persian_memory.py
"""
import base64
import io
import hashlib
import json
import logging
import re
import sqlite3
import threading
import webbrowser
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import load_config
from app.db.backup import backup_database as create_database_backup
from app.db.backup import database_status as get_database_status
from app.db.connection import connect as database_connect
from app.db.migrations import SCHEMA_VERSION, apply_migrations as run_migrations
from app.db.migrations import initialize_database
from app.db.repositories import Repositories
from app.services.diagnostics import embedding_diagnostics
from app.services.chat_service import ChatService
from app.services.ollama_client import OllamaClient
from app.services.prompt_builder import PromptBuilder

BASE = Path(__file__).resolve().parent
CONFIG = load_config(BASE)
DATA = CONFIG.data_directory
DATA.mkdir(parents=True, exist_ok=True)
DB = CONFIG.database_path
HOST = CONFIG.host
PORT = CONFIG.port
OLLAMA = CONFIG.ollama_url
MAX_REQUEST = CONFIG.max_request_bytes
MAX_ATTACHMENT_CHARS = CONFIG.max_attachment_chars
MAX_CONTEXT = CONFIG.max_context_chars
EMBED_MODEL = CONFIG.embedding_model
MAX_DOC_SIZE = CONFIG.max_document_bytes
MAX_DOC_CHARS = CONFIG.max_document_chars
# OOXML files are ZIP archives.  Limit their total declared uncompressed size,
# not just individual XML members, so a small compressed upload cannot cause a
# large allocation during local document extraction.
MAX_ARCHIVE_MEMBERS = CONFIG.max_archive_members
MAX_ARCHIVE_UNCOMPRESSED = CONFIG.max_archive_uncompressed_bytes
LOG_MAX_BYTES = CONFIG.log_max_bytes
LOG_BACKUP_COUNT = CONFIG.log_backup_count
ALLOWED_CONTEXT_WINDOWS = CONFIG.allowed_context_windows
def connect():
    """Compatibility wrapper for the established module-level database path."""
    return database_connect(DB)


def configure_logging():
    """Configure local operational logging without request or document content."""
    logger = logging.getLogger('offline_persian_assistant')
    target = str(DATA / 'assistant.log')
    for handler in logger.handlers:
        if getattr(handler, '_assistant_log_path', None) == target:
            return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    DATA.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(target, maxBytes=LOG_MAX_BYTES,
                                  backupCount=LOG_BACKUP_COUNT, encoding='utf-8')
    handler._assistant_log_path = target
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_event(level, event, **details):
    """Write structured metadata only; never pass content, prompts, or credentials."""
    getattr(configure_logging(), level)(json.dumps(
        {'event': event, **details}, ensure_ascii=False, sort_keys=True))


def repositories():
    """Create a repository facade bound to the existing configured database."""
    return Repositories(connect)


def apply_migrations(database):
    """Compatibility wrapper for callers using the previous module function."""
    return run_migrations(database, log_event)


def initialize():
    initialize_database(connect, log_event)


def database_status():
    return get_database_status(connect)


def backup_database(destination=None):
    return create_database_backup(connect, DATA, destination, log_event)


def ollama_client(tags_timeout=None):
    """Create a local Ollama client from the compatibility configuration values."""
    return OllamaClient(
        OLLAMA,
        tags_timeout=CONFIG.ollama_tags_timeout if tags_timeout is None else tags_timeout,
        embedding_timeout=CONFIG.ollama_embedding_timeout,
        chat_timeout=CONFIG.ollama_chat_timeout)


def ollama_status():
    return {'connected': ollama_client(CONFIG.ollama_health_timeout).is_available()}


def embedding_status():
    return embedding_diagnostics(ollama_client(), EMBED_MODEL)


def health_status():
    try:
        database = database_status()
    except sqlite3.Error:
        log_event('error', 'database_health_failed', error_type='sqlite_error')
        database = {'connected': False}
    return {
        'ollama': ollama_status(),
        'embedding': embedding_status(),
        'database': database,
    }


def norm(text):
    normalized = str(text).lower().replace('ي', 'ی').replace('ك', 'ک').replace('‌', ' ')
    # Diacritics are optional in Persian writing; retaining them causes the
    # lexical fallback to miss otherwise identical words.
    normalized = re.sub(r'[\u064b-\u065f\u0670]', '', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()


def search_knowledge(question, limit=5):
    """Simple offline lexical retrieval; no embedding / semantic search yet."""
    terms = [t for t in re.findall(r'[\w\u0600-\u06ff]{3,}', norm(question))
             if t not in {'برای', 'چگونه', 'درباره', 'میخواهم', 'میشود', 'چیست', 'کنید', 'لطفا', 'اینکه', 'همه', 'اطلاعات', 'شرکت', 'دارد'}]
    terms = list(dict.fromkeys(terms))[:24]
    rows = repositories().approved_knowledge()
    scored = []
    for r in rows:
        title, content, category = norm(r['title']), norm(r['content']), norm(r['category'])
        score = sum((5 if t in title else 0) + (2 if t in category else 0) + (1 if t in content else 0) for t in terms)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda s:(-s[0], s[1]['id']))
    # If there were no meaningful matching terms, do not inject unrelated company facts.
    return [dict(r) for _, r in scored[:limit]]


PERSIAN_STOP = {'این','آن','های','برای','است','هست','باشد','شده','درباره','چگونه','چه','چرا','شرکت','لطفا','لطفاً','بگو','کن','کنید','را','از','در','به','با','یا','و','که','یک','آیا','اطلاعات'}

def tokenize(text):
    return set(t for t in re.findall(r'[\w\u0600-\u06ff]{3,}',norm(text)) if t not in PERSIAN_STOP)

def chunk_document(text, chunk_chars=1300, overlap=180):
    """Approximate character-based chunks with overlap, retaining paragraph flow."""
    text = text.strip()
    if not text:return []
    chunks=[];pos=0
    while pos < len(text):
        end=min(pos+chunk_chars,len(text))
        if end<len(text):
            pivot=max(text.rfind('\n',pos+chunk_chars//2,end),text.rfind(' ',pos+chunk_chars//2,end))
            if pivot>pos+chunk_chars//2:end=pivot+1
        chunks.append(text[pos:end].strip())
        if end>=len(text):break
        pos=max(pos+1,end-overlap)
    return [x for x in chunks if x]

def ollama_embedding(text, model):
    return ollama_client().embed(text, model)


def chat_service():
    """Create the compatibility chat coordinator for the current application wiring."""
    return ChatService(repositories, ollama_client, search_knowledge, search_documents,
                       PromptBuilder(), ALLOWED_CONTEXT_WINDOWS)

def ingest_document(name, raw, model):
    if not name or len(name)>260:raise ValueError('نام فایل نامعتبر است')
    if len(raw)>MAX_DOC_SIZE:raise ValueError('حجم سند بیشتر از ۸ مگابایت است')
    name=Path(name.replace('\\','/')).name
    if not name or name in ('.', '..'):
        raise ValueError('نام فایل نامعتبر است')
    text=document_text(name,raw)
    if not text.strip():raise ValueError('متنی در سند یافت نشد؛ PDF اسکن‌شده به OCR نیاز دارد')
    if len(text)>MAX_DOC_CHARS:raise ValueError('متن سند بیش از حد بزرگ است؛ آن را به چند فایل تقسیم کنید')
    digest=hashlib.sha256(raw).hexdigest()
    row = repositories().find_document_by_digest(digest)
    if row:raise ValueError('این فایل قبلاً وارد پایگاه دانش شده است (شناسه '+str(row['id'])+')')
    chunks=chunk_document(text)
    if len(chunks)>1000:raise ValueError('تعداد بخش‌های سند بیش از حد زیاد است')
    vectors=[]
    if model:
        for part in chunks:vectors.append(ollama_embedding(part,model))
    else:vectors=[None for _ in chunks]
    stored_chunks = [(index + 1, part, json.dumps(vector) if vector else '')
                     for index, (part, vector) in enumerate(zip(chunks, vectors))]
    ident = repositories().create_document_with_chunks(
        name, digest, len(text), stored_chunks, model,
        datetime.now().isoformat(timespec='seconds'))
    return {'id':ident,'name':name,'chunks':len(chunks),'mode':'semantic' if model else 'lexical'}

def search_documents(query,limit=5):
    """Hybrid cosine/term scoring; semantic only for vectors indexed with selected model."""
    records = repositories().document_records()
    words=tokenize(query)
    model=EMBED_MODEL
    qvec=None
    if model and any(r['embed_model']==model and r['vector'] for r in records):
        try:qvec=ollama_embedding(query,model)
        except Exception:pass # Safe lexical fallback if embedding service unavailable.
    ranked=[]
    for r in records:
        body=norm(r['content']);title=norm(r['name'])
        hits=sum(2 if t in title else 0 for t in words)+sum(1 for t in words if t in body)
        lex=hits/max(1,len(words))
        sem=None
        if qvec is not None and r['embed_model']==model and r['vector']:
            try:
                vec=json.loads(r['vector'])
                if len(vec)==len(qvec):sem=sum(a*b for a,b in zip(qvec,vec))
            except (ValueError,TypeError):pass
        if sem is None:
            if not hits:continue
            score=lex
        else:
            # Skip weak semantic matches without keyword overlap.
            if sem<0.38 and not hits:continue
            score=1.8*sem+0.5*min(lex,3)
        ranked.append((score,dict(r)))
    ranked.sort(key=lambda v:-v[0])
    return [r for _,r in ranked[:limit]]


def document_text(name, content):
    ext = Path(name).suffix.lower()
    if ext in ('.txt', '.md', '.csv', '.json', '.log'):
        for enc in ('utf-8-sig', 'utf-16', 'cp1256'):
            try:
                return content.decode(enc)
            except UnicodeError:
                pass
        return content.decode('utf-8', errors='replace')
    if ext == '.docx':
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            validate_office_archive(z)
            if 'word/document.xml' not in z.namelist() or z.getinfo('word/document.xml').file_size > 20_000_000:
                raise ValueError('سند Word بیش از حد بزرگ است')
            root = ET.fromstring(z.read('word/document.xml'))
        ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        return '\n'.join(''.join(t.text or '' for t in p.iter(ns+'t')) for p in root.iter(ns+'p'))
    if ext == '.xlsx':
        ns='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            validate_office_archive(z)
            shared=[]
            if 'xl/sharedStrings.xml' in z.namelist():
                if z.getinfo('xl/sharedStrings.xml').file_size > 20_000_000:
                    raise ValueError('فایل Excel بیش از حد بزرگ است')
                s=ET.fromstring(z.read('xl/sharedStrings.xml'))
                shared=[''.join(t.text or '' for t in si.iter(ns+'t')) for si in s.iter(ns+'si')]
            lines=[]
            for name in sorted(n for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml',n)):
                if z.getinfo(name).file_size > 20_000_000:
                    raise ValueError('برگه Excel بیش از حد بزرگ است')
                root=ET.fromstring(z.read(name));lines.append('['+name+']')
                for row in root.iter(ns+'row'):
                    cells=[]
                    for c in row.iter(ns+'c'):
                        v=c.find(ns+'v'); val=v.text if v is not None and v.text else ''
                        if c.get('t')=='s' and val.isdigit() and int(val)<len(shared):
                            val=shared[int(val)]
                        if c.get('t')=='inlineStr':
                            val=''.join(t.text or '' for t in c.iter(ns+'t'))
                        if val:cells.append(c.get('r','')+'='+val)
                    if cells:lines.append(' | '.join(cells))
            return '\n'.join(lines)
    if ext == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ValueError('برای PDF متنی کتابخانه pypdf را آفلاین نصب کنید') from e
        return '\n'.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(content)).pages)
    raise ValueError('فرمت فایل پشتیبانی نمی‌شود')


def validate_office_archive(archive):
    """Reject oversized OOXML ZIPs before reading XML into memory."""
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_MEMBERS or sum(info.file_size for info in infos) > MAX_ARCHIVE_UNCOMPRESSED:
        raise ValueError('محتوای فشرده سند بیش از حد بزرگ است')


PAGE = r'''<!doctype html><html lang="fa" dir="rtl"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>دستیار سازمانی آفلاین</title>
<style>
:root{font-family:Tahoma,"Segoe UI",sans-serif;color:#1c2940;background:#f1f4f8;font-size:14px}*{box-sizing:border-box}body{margin:0;line-height:1.9}
.app{display:grid;grid-template-columns:270px minmax(0,1fr);height:100dvh;max-width:1600px;margin:auto;background:white}.side{background:#f8fafc;border-left:1px solid #e2e8f0;padding:15px;overflow:auto}h2{font-size:17px;margin:4px 0 15px}h3{font-size:14px;margin:20px 0 6px}.side button{width:100%;margin:4px 0;text-align:right}.chat-item{padding:9px;border-radius:8px;cursor:pointer;overflow-wrap:anywhere}.chat-item:hover,.chat-item.active{background:#e3edf9}.main{display:flex;min-height:0;flex-direction:column}.top{padding:13px 18px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;gap:10px;flex-wrap:wrap}.top strong{margin-left:auto}.feed{flex:1;min-height:0;overflow:auto;padding:20px;display:flex;flex-direction:column;gap:15px}.msg{max-width:90%;align-self:flex-start}.msg.user{align-self:flex-end}.role{color:#64748b;font-size:12px;font-weight:bold}.bubble{white-space:pre-wrap;unicode-bidi:plaintext;text-align:start;overflow-wrap:anywhere;background:#f3f5f9;border-radius:13px;padding:11px 14px}.user .bubble{background:#e4efff}.composer{border-top:1px solid #e2e8f0;padding:14px}textarea,input,select{font-family:inherit;font-size:14px;border:1px solid #cad5e3;border-radius:8px;padding:8px;color:#1c2940}textarea{width:100%;min-height:85px;resize:vertical;direction:rtl;text-align:right}button{font-family:inherit;cursor:pointer;padding:8px 13px;background:#2764bd;color:white;border:0;border-radius:8px}button:disabled{opacity:.5;cursor:wait}.subtle{background:#e9eef7;color:#244a82}.danger{background:#fff0f0;color:#9c3434}.actions{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-top:7px}.hint{color:#66758b;font-size:12px}.knowledge{display:none;position:fixed;inset:0;background:#0f172a8c;align-items:center;justify-content:center;padding:12px;z-index:4}.panel{width:min(900px,100%);max-height:93vh;overflow:auto;background:white;border-radius:14px;padding:20px}.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.formgrid label{display:flex;flex-direction:column;gap:4px}.formgrid .wide{grid-column:1/-1}.formgrid textarea{min-height:110px}.fact{border-bottom:1px solid #e2e8f0;padding:10px 0;overflow-wrap:anywhere}.fact .meta{font-size:12px;color:#66758b}.think{background:#f8fafc;border:1px solid #e2e8f0;padding:8px 12px;border-radius:10px}.think summary{cursor:pointer;color:#64748b}.think pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;max-height:230px;overflow:auto}.sources{font-size:12px;color:#4b6587;margin:5px 0}.status{font-size:12px;color:#64748b}
@media(max-width:740px){.app{grid-template-columns:1fr}.side{max-height:155px;border-left:0;border-bottom:1px solid #e2e8f0;padding:8px}.side h3{display:none}.chat-item{display:inline-block;margin:3px;font-size:12px}.side button{width:auto}.feed{padding:12px}.msg{max-width:98%}.formgrid{grid-template-columns:1fr}.formgrid .wide{grid-column:auto}}
</style></head><body><div class="app"><aside class="side"><h2>دستیار سازمانی</h2><button id="newChat">+ گفت‌وگوی جدید</button><button class="subtle" id="openKnowledge">مدیریت دانش شرکت</button><button class="subtle" id="openDocuments">اسناد و RAG</button><h3>گفت‌وگوهای ذخیره‌شده</h3><div id="chatList"></div></aside><section class="main"><div class="top"><strong>Gemma / Ollama • آفلاین</strong><span>مدل:</span><select id="model"><option>در حال اتصال...</option></select><span>Context:</span><select id="ctx"><option value="4096">4K</option><option value="8192" selected>8K</option><option value="16384">16K</option><option value="32768">32K</option></select><button id="refresh" class="subtle">بازخوانی</button></div><div class="feed" id="feed"><div class="hint">سلام! اطلاعات تأییدشده شرکت را در «مدیریت دانش شرکت» ثبت کنید. سپس درباره آن‌ها سؤال بپرسید.</div></div><div class="composer"><textarea id="prompt" placeholder="سؤال خود را به فارسی بنویسید... (Enter: ارسال، Shift+Enter: سطر جدید)"></textarea><div class="actions"><div><input type="file" id="file" accept=".txt,.md,.csv,.json,.log,.docx,.xlsx,.pdf" style="max-width:220px"><button id="removeFile" class="subtle">حذف فایل</button></div><div><span class="status" id="status">آماده</span> <button id="send">ارسال</button></div></div><div class="hint">ذخیره خودکار گفت‌وگوها فعال است. دانش جدید فقط با ثبت و تأیید شما وارد حافظه سازمانی می‌شود. جست‌وجوی ترکیبی اسناد با مدل Embedding اختیاری فعال است.</div></div></section></div>
<div class="knowledge" id="knowledgeModal"><div class="panel"><div class="actions"><h2>مدیریت دانش سازمانی</h2><button id="closeKnowledge" class="subtle">بستن</button></div><div class="formgrid"><label>عنوان<input id="factTitle" maxlength="180" placeholder="مثلاً ساختار کمیته پروژه"></label><label>دسته‌بندی<input id="factCategory" maxlength="100" value="عمومی"></label><label class="wide">متن دانش تأییدشده<textarea id="factContent" placeholder="قاعده، رویه یا اطلاعات مصوب را بنویسید..."></textarea></label><label class="wide">منبع (نام سند، شماره مصوبه و...)<input id="factSource" maxlength="300" placeholder="اختیاری"></label></div><div class="actions"><span class="hint">اطلاعات صرفاً با انتخاب «ثبت دانش تأییدشده» ذخیره می‌شوند.</span><div><button id="cancelEdit" class="subtle">پاک کردن فرم</button> <button id="saveFact">ثبت دانش تأییدشده</button></div></div><p id="factStatus" class="status"></p><h3>دانش ذخیره‌شده</h3><div id="factList"></div></div></div>
<div class="knowledge" id="docsModal"><div class="panel"><div class="actions"><h2>اسناد پایگاه دانش (RAG)</h2><button class="subtle" id="closeDocuments">بستن</button></div><p class="hint">فایل‌های PDF متنی، Word، Excel و TXT را برای استفاده در گفت‌وگوهای آینده ثبت کنید. بدون مدل Embedding، جست‌وجو به صورت واژگانی انجام می‌شود.</p><input type="file" id="indexFile" accept=".txt,.md,.csv,.json,.log,.docx,.xlsx,.pdf"><button id="indexDocument">افزودن سند به پایگاه دانش</button><p id="docStatus" class="status"></p><h3>اسناد ثبت‌شده</h3><div id="docList"></div></div></div>
<script>
'use strict';const $=id=>document.getElementById(id);let chatId=null,busy=false,editing=null;
async function api(path,payload){const opts=payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)};const r=await fetch(path,opts);const data=await r.json().catch(()=>({error:'پاسخ نامعتبر'}));if(!r.ok)throw Error(data.error||'خطای سرور');return data}
function message(role,content){const box=document.createElement('div');box.className='msg '+role;const label=document.createElement('div');label.className='role';label.textContent=role==='user'?'شما':'Gemma 4';const bubble=document.createElement('div');bubble.className='bubble';bubble.dir='auto';bubble.textContent=content;box.append(label,bubble);$('feed').append(box);$('feed').scrollTop=$('feed').scrollHeight;return bubble}
function busyState(on,label){busy=on;$('send').disabled=on;$('status').textContent=label||'آماده'}
async function models(){try{const res=await api('/api/models');const list=res.models||[];const old=$('model').value;$('model').replaceChildren();list.forEach(m=>{const o=document.createElement('option');o.textContent=m;o.value=m;$('model').append(o)});$('model').value=list.includes(old)?old:(list.find(x=>/gemma/i.test(x))||list[0]||'');$('status').textContent='اتصال Ollama برقرار است'}catch(e){$('status').textContent='خطای Ollama: '+e.message}}
async function loadChats(){const res=await api('/api/chats');$('chatList').replaceChildren();res.chats.forEach(c=>{const el=document.createElement('div');el.className='chat-item'+(c.id===chatId?' active':'');el.textContent=c.title;el.onclick=()=>openChat(c.id);$('chatList').append(el)})}
async function openChat(id){if(busy)return;const res=await api('/api/chat-history?id='+encodeURIComponent(id));chatId=id;$('feed').replaceChildren();res.messages.forEach(x=>message(x.role,x.content));await loadChats()}
async function newChat(){if(busy)return;chatId=null;$('feed').replaceChildren();const msg=document.createElement('div');msg.className='hint';msg.textContent='گفت‌وگوی جدید؛ پیام خود را بنویسید.';$('feed').append(msg);await loadChats()}
$('newChat').onclick=newChat;$('refresh').onclick=models;
async function readFile(){const f=$('file').files[0];if(!f)return null;if(f.size>8*1024*1024)throw Error('فایل نباید بیشتر از ۸ مگابایت باشد');const buf=new Uint8Array(await f.arrayBuffer());let binary='';for(let i=0;i<buf.length;i+=4096)binary+=String.fromCharCode(...buf.subarray(i,i+4096));return {name:f.name,data:btoa(binary)}}
$('removeFile').onclick=()=>{$('file').value=''};
async function send(){if(busy)return;const question=$('prompt').value.trim();if(!question)return;if(!$('model').value){alert('ابتدا مدل Ollama را انتخاب کنید');return}busyState(true,'در حال آماده‌سازی...');let bubble,thinkBox;
 try{let content=question;const f=await readFile();if(f){const doc=await api('/api/extract',f);if(!doc.text.trim())throw Error('متنی در سند یافت نشد');content+='\n\n[سند ضمیمه: '+f.name+']\n'+doc.text+(doc.truncated?'\n[سند کوتاه شده است]':'')}
 message('user',question+(f?'\n📎 '+f.name:''));bubble=message('assistant','در حال بررسی دانش سازمانی...');const r=await fetch('/api/chat-stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({chat_id:chatId,question,content,model:$('model').value,ctx:Number($('ctx').value)})});if(!r.ok){const error=await r.json().catch(()=>({}));throw Error(error.error||'خطای ارسال')}
 if(!r.body)throw Error('مرورگر پخش جریانی را پشتیبانی نمی‌کند');const reader=r.body.getReader(),decoder=new TextDecoder();let buffer='',full='',thinking='';bubble.textContent='';
 function event(line){if(!line.trim())return;const data=JSON.parse(line);if(data.error)throw Error(data.error);if(data.chat_id)chatId=data.chat_id;
 if((data.source_ids?.length||data.doc_sources?.length)&&!bubble.dataset.sources){const sources=document.createElement('div');sources.className='sources';sources.textContent='منابع: '+[...(data.source_ids||[]).map(x=>'K'+x),...(data.doc_sources||[])].join('، ');bubble.parentNode.insertBefore(sources,bubble);bubble.dataset.sources='1'}
 if(data.thinking){if(!thinkBox){thinkBox=document.createElement('details');thinkBox.className='think';thinkBox.open=true;const title=document.createElement('summary');title.textContent='خروجی Thinking ارائه‌شده توسط مدل';const pre=document.createElement('pre');thinkBox.append(title,pre);bubble.parentElement.insertBefore(thinkBox,bubble)}thinking+=data.thinking;thinkBox.querySelector('pre').textContent=thinking;$('status').textContent='مدل در حال فکر کردن است...'}
 if(data.content){full+=data.content;bubble.textContent=full;$('status').textContent='در حال نوشتن پاسخ...'}if(data.done){if(thinkBox)thinkBox.open=false;$('status').textContent='پاسخ کامل شد'}$('feed').scrollTop=$('feed').scrollHeight}
 while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines)event(line)}buffer+=decoder.decode();if(buffer.trim())event(buffer);
 if(!full)bubble.textContent='پاسخ متنی دریافت نشد';$('prompt').value='';$('file').value='';await loadChats();busyState(false,'آماده')
 }catch(e){if(bubble)bubble.textContent='خطا: '+e.message;else alert(e.message);busyState(false,'خطا: '+e.message)}}
$('send').onclick=send;$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send()}});
function clearForm(){editing=null;$('factTitle').value='';$('factContent').value='';$('factSource').value='';$('factCategory').value='عمومی';$('saveFact').textContent='ثبت دانش تأییدشده'}
async function facts(){const res=await api('/api/knowledge');$('factList').replaceChildren();res.facts.forEach(f=>{const box=document.createElement('div');box.className='fact';const b=document.createElement('b');b.textContent='#'+f.id+' • '+f.title;const meta=document.createElement('div');meta.className='meta';meta.textContent='دسته: '+f.category+' | منبع: '+(f.source||'ثبت دستی')+' | ویرایش: '+f.updated_at;const para=document.createElement('div');para.textContent=f.content;const edit=document.createElement('button');edit.className='subtle';edit.textContent='ویرایش';edit.onclick=()=>{editing=f.id;$('factTitle').value=f.title;$('factContent').value=f.content;$('factCategory').value=f.category;$('factSource').value=f.source;$('saveFact').textContent='ذخیره تغییرات تأییدشده';$('factTitle').focus()};const del=document.createElement('button');del.className='danger';del.style.marginRight='8px';del.textContent='حذف';del.onclick=async()=>{if(!confirm('این دانش از پایگاه اطلاعات حذف شود؟'))return;try{await api('/api/knowledge-delete',{id:f.id});await facts()}catch(e){alert(e.message)}};box.append(b,meta,para,edit,del);$('factList').append(box)})}
$('openKnowledge').onclick=async()=>{$('knowledgeModal').style.display='flex';try{await facts()}catch(e){$('factStatus').textContent=e.message}};$('closeKnowledge').onclick=()=>{$('knowledgeModal').style.display='none'};$('cancelEdit').onclick=clearForm;
$('saveFact').onclick=async()=>{try{const payload={id:editing,title:$('factTitle').value.trim(),category:$('factCategory').value.trim(),content:$('factContent').value.trim(),source:$('factSource').value.trim()};await api('/api/knowledge-save',payload);clearForm();$('factStatus').textContent='دانش تأییدشده ذخیره شد';await facts()}catch(e){$('factStatus').textContent='خطا: '+e.message}};
$('openDocuments').onclick=async()=>{$('docsModal').style.display='flex';await loadDocs()};
$('closeDocuments').onclick=()=>{$('docsModal').style.display='none'};
async function loadDocs(){try{const res=await api('/api/documents');$('docList').replaceChildren();res.documents.forEach(d=>{const line=document.createElement('div');line.className='fact';const title=document.createElement('b');title.textContent='#'+d.id+' • '+d.name;const meta=document.createElement('div');meta.className='meta';meta.textContent=d.chunks+' بخش • '+(d.embed_model?'Embedding: '+d.embed_model:'جست‌وجوی متنی');const del=document.createElement('button');del.className='danger';del.textContent='حذف سند';del.onclick=async()=>{if(confirm('سند و بخش‌های آن حذف شوند؟')){await api('/api/document-delete',{id:d.id});await loadDocs()}};line.append(title,meta,del);$('docList').append(line)})}catch(e){$('docStatus').textContent=e.message}}
$('indexDocument').onclick=async()=>{const f=$('indexFile').files[0];if(!f){$('docStatus').textContent='ابتدا فایل را انتخاب کنید';return}const btn=$('indexDocument');btn.disabled=true;$('docStatus').textContent='در حال استخراج متن و تولید Embedding؛ برای فایل بزرگ زمان بیشتری لازم است...';try{if(f.size>8*1024*1024)throw Error('حجم فایل بیش از ۸ مگابایت است');const arr=new Uint8Array(await f.arrayBuffer());let binary='';for(let i=0;i<arr.length;i+=4096)binary+=String.fromCharCode(...arr.subarray(i,i+4096));const res=await api('/api/document-add',{name:f.name,data:btoa(binary)});$('docStatus').textContent='سند #'+res.id+' در '+res.chunks+' بخش ذخیره شد ('+res.mode+')';$('indexFile').value='';await loadDocs()}catch(e){$('docStatus').textContent='خطا: '+e.message}finally{btn.disabled=false}};
models();loadChats();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def json_response(self, result, code=200):
        raw=json.dumps(result,ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        try:
            path=self.path.split('?',1)[0]
            if path=='/':
                raw=PAGE.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'")
                self.send_header('Content-Length',str(len(raw)))
                self.end_headers();self.wfile.write(raw)
            elif path=='/api/health':
                self.json_response(health_status())
            elif path=='/api/models':
                self.json_response({'models':ollama_client().list_models()})
            elif path=='/api/documents':
                rows = repositories().list_documents()
                self.json_response({'documents':[dict(x) for x in rows]})
            elif path=='/api/knowledge':
                rows = repositories().list_knowledge()
                self.json_response({'facts':[dict(x) for x in rows]})
            elif path=='/api/chats':
                rows = repositories().list_chats()
                self.json_response({'chats':[dict(x) for x in rows]})
            elif path=='/api/chat-history':
                from urllib.parse import parse_qs,urlsplit
                ident=parse_qs(urlsplit(self.path).query).get('id',[''])[0]
                if not re.fullmatch(r'[a-f0-9]{32}',ident):raise ValueError('شناسه گفت‌وگو نامعتبر است')
                rows = repositories().chat_history(ident)
                self.json_response({'messages':[dict(x) for x in rows]})
            else:self.json_response({'error':'مسیر نامعتبر است'},404)
        except Exception as exc:
            log_event('error', 'api_get_failed', error_type=type(exc).__name__, path=path)
            self.json_response({'error':'خطای داخلی سرور'},502)

    def do_POST(self):
        if self.path not in ('/api/knowledge-save','/api/knowledge-delete','/api/chat-stream','/api/extract','/api/document-add','/api/document-delete'):
            self.json_response({'error':'مسیر نامعتبر است'},404);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>MAX_REQUEST:raise ValueError('حجم درخواست نامعتبر است')
            obj=json.loads(self.rfile.read(length))
            if not isinstance(obj,dict):raise ValueError('درخواست نامعتبر است')
            if self.path=='/api/document-add':
                name=str(obj.get('name',''))
                raw=base64.b64decode(obj['data'],validate=True)
                self.json_response(ingest_document(name,raw,EMBED_MODEL))
            elif self.path=='/api/document-delete':
                repositories().delete_document(obj['id'])
                self.json_response({'deleted':True})
            elif self.path=='/api/knowledge-save':
                title=str(obj.get('title','')).strip();category=str(obj.get('category','عمومی')).strip() or 'عمومی';content=str(obj.get('content','')).strip();source=str(obj.get('source','')).strip()
                if not title or not content or len(title)>180 or len(category)>100 or len(content)>10000 or len(source)>300:
                    raise ValueError('عنوان/متن دانش خالی یا بیش از حد طولانی است')
                now=datetime.now().isoformat(timespec='seconds')
                ident = repositories().save_knowledge(
                    obj.get('id'), title, category, content, source, now)
                self.json_response({'id':ident,'saved':True})
            elif self.path=='/api/knowledge-delete':
                repositories().delete_knowledge(obj['id'])
                self.json_response({'deleted':True})
            elif self.path=='/api/extract':
                name=str(obj.get('name',''));raw=base64.b64decode(obj['data'],validate=True)
                if len(raw)>8*1024*1024:raise ValueError('فایل بیش از حد بزرگ است')
                output=document_text(name,raw)
                self.json_response({'text':output[:MAX_ATTACHMENT_CHARS],'truncated':len(output)>MAX_ATTACHMENT_CHARS})
            else:self.chat_stream(obj)
        except (ValueError,TypeError,KeyError,UnicodeError,zipfile.BadZipFile,ET.ParseError) as e:
            self.json_response({'error':str(e)},400)
        except Exception as e:
            log_event('error', 'api_post_failed', error_type=type(e).__name__, path=self.path)
            self.json_response({'error':'خطای داخلی سرور'},502)

    def chat_stream(self,obj):
        session = chat_service().start(obj)
        diagnostics = session.diagnostics
        if diagnostics.truncated_sections:
            log_event('info', 'prompt_truncated',
                      section_count=len(diagnostics.truncated_sections),
                      estimated_chars=diagnostics.estimated_input_chars,
                      budget_chars=diagnostics.input_budget_chars)
        self.send_response(200)
        self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control','no-cache, no-transform')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        def emit(payload):
            self.wfile.write((json.dumps(payload,ensure_ascii=False)+'\n').encode('utf-8'))
            self.wfile.flush()
        try:
            for event in session.events():
                emit(event)
        except (BrokenPipeError,ConnectionResetError):
            log_event('info', 'stream_interrupted')
        except Exception as e:
            log_event('error', 'stream_failed', error_type=type(e).__name__)
            try:emit({'error':'خطا در دریافت پاسخ از Ollama'})
            except (BrokenPipeError,ConnectionResetError):pass


def main():
    initialize()
    try:server=ThreadingHTTPServer((HOST,PORT),Handler)
    except OSError as e:
        print('پورت در دسترس نیست. نسخه قبلی را ببندید یا OLLAMA_UI_PORT را تغییر دهید:',e)
        return
    url=f'http://{HOST}:{PORT}/'
    print('Offline Persian Company Assistant:',url)
    print('SQLite database:',DB)
    print('Stop with Ctrl+C')
    threading.Timer(0.7,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
