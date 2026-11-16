# ارزیابی معماری و نقشهٔ راه

## وضعیت موجود و جریان کامل

`ollama_persian_rag.py` یک HTTP server استاندارد Python، HTML/JS درون‌خطی و SQLite را در یک فایل گرد هم می‌آورد.

1. رابط RTL درخواست را به `/api/chat-stream` می‌فرستد؛ server دانش تأییدشده و chunkهای سند را بازیابی می‌کند، history را می‌خواند، سپس `/api/chat` محلی Ollama را به NDJSON مرورگر stream می‌کند. فقط پس از پاسخ کامل، پیام کاربر و پاسخ در `messages` ذخیره می‌شوند.
2. دانش دستی در `knowledge` ثبت، و با tokenهای نرمال‌شده به‌صورت lexical رتبه‌بندی می‌شود.
3. سند با base64 می‌رسد، متن TXT/DOCX/XLSX/PDF استخراج، chunk و هش می‌شود و در `documents` و `doc_chunks` ذخیره می‌گردد. اگر `OLLAMA_EMBED_MODEL` تعیین شده باشد، embedding هر chunk از Ollama محلی دریافت می‌شود؛ در غیر این صورت lexical fallback فعال است.
4. RAG برای مدل embedding دقیقاً هم‌نام، cosine و lexical score را ترکیب می‌کند و citationهای `K#` و `D#-C#` به prompt و UI می‌افزاید.

نکات مثبت: bind محلی، SQL پارامتری، CSP پایه، استفاده از `textContent`، stream، محدودیت اندازه و citation اولیه وجود دارد. RTL پایه نیز درست است: `lang=fa dir=rtl`، textarea RTL، و پیام‌ها `unicode-bidi: plaintext` دارند.

## مسائل اولویت‌دار

| اولویت | مسئله | اثر / اقدام |
|---|---|---|
| P0 | کد، HTML، DB و منطق Ollama در یک فایل‌اند | جداسازی مرحله‌ای با حفظ endpointها و قرارداد DB |
| P0 | migration versioned، backup/restore و integrity check وجود ندارد | migration افزایشی، `PRAGMA user_version`، backup امن و آزمون migration |
| P0 | پاسخ نیمه‌کاره ذخیره نمی‌شود و خطای stream audit ندارد | وضعیت پیام/درخواست، logging محلی چرخشی و آزمون قطع اتصال |
| P1 | بودجهٔ prompt بر مبنای کاراکتر است و `MAX_CONTEXT` استفاده نمی‌شود | prompt builder با budget صریح و آزمون overflow |
| P1 | embedding در مسیر درخواست و به‌صورت ترتیبی است | job state محلی، diagnostics مدل/ابعاد، retry محدود و cancellation |
| P1 | بازیابی lexical با substring و scan کامل DB است | tokenization فارسی آزموده، metadata/citation، FTS یا index محلی پس از migration |
| P1 | PDF فقط متن داخلی؛ جدول/شماره صفحه/source span حفظ نمی‌شود | extractor interface، page citations و OCR محلی اختیاری |
| P1 | delete سند فوراً دائمی است | تأیید server-side، audit و backup/restore پیش از UI بهبود |
| P2 | UI embedded و accessibility/keyboard/error states محدود است | دارایی‌های جدا، RTL visual tests، وضعیت stream و citation قابل‌کلیک |
| P2 | هیچ auth محلی، نقش یا سیاست نگهداری نیست | ابتدا threat model؛ اگر multi-user شد، auth محلی و RBAC |

## بهبودهای ایمن این مرحله

- foreign key enforcement برای همهٔ connectionها فعال شد.
- نرمال‌سازی retrieval، اعراب عربی/فارسی را حذف می‌کند تا fallback واژگانی کمتر miss کند.
- DOCX/XLSX علاوه بر اندازهٔ عضو، سقف تعداد عضو و مجموع uncompressed دارند.
- چهار آزمون استاندارد-library با DB و فایل‌های ساختگی افزوده شدند.

## ساختار پیشنهادی (هدف، نه مهاجرت فوری)

```text
app/
  config.py  db/{connection.py,migrations.py,repositories.py}
  services/{chat.py,ollama.py,retrieval.py,ingestion.py,extraction.py}
  web/{server.py,handlers.py,static/}
  domain/{models.py,errors.py}
tests/{test_retrieval.py,test_ingestion.py,test_api.py,test_migrations.py}
scripts/{backup.py,verify_db.py}
```

ابتدا wrapperهای سازگار پیرامون تابع‌های فعلی بسازید، سپس هر endpoint را یکی‌یکی انتقال دهید. schema فعلی (`knowledge`, `chats`, `messages`, `documents`, `doc_chunks`) قرارداد داده است و نباید بازنویسی یا drop شود.

## فازها

1. **Stabilize:** test matrix، logging امن، health check، backup/verify script و migration framework افزایشی.
2. **Modularize:** config/DB/repository/service/web را پشت API فعلی استخراج کنید؛ API streaming و schema را ثابت نگه دارید.
3. **RAG diagnostics:** ثبت مدل/بعد/vector version، re-index صریح و opt-in، quality set فارسی و citation/page span.
4. **Document lifecycle:** ingestion job، وضعیت/خطا، preview، archive/restore و extractorهای قابل‌آزمون.
5. **Read-only SQL Server:** connector اختیاری، credential خارج از Git، network allowlist، query templates، timeout/row limit/audit. هیچ SQL آزاد از prompt اجرا نشود.
6. **UI و عملیات:** UI حرفه‌ای RTL، accessibility، access control متناسب با threat model، backup، retention، offline Windows packaging.

## ریسک‌ها و تصمیم‌های باز

- نسخهٔ دقیق Ollama/Gemma، ظرفیت سخت‌افزار و مدل embedding مجاز مشخص نیست؛ re-index خودکار ممنوع بماند.
- تغییر `foreign_keys=ON` ممکن است دادهٔ تاریخی orphan را هنگام عملیات آینده آشکار کند؛ پیش از migration با `PRAGMA foreign_key_check` گزارش گرفته شود.
- CSP فعلی به inline script/style نیاز دارد؛ هنگام جداکردن UI باید nonce/hash جایگزین شود.
- ورودی محلی بدون auth برای کاربر واحد مناسب است، نه workstation مشترک یا شبکه؛ دامنهٔ استقرار باید تعیین شود.
- استخراج PDFهای اسکن‌شده، XLSXهای پیچیده (formula/date/style) و DOCXهای دارای جدول/تصویر پوشش کامل ندارند.
- citation نشان می‌دهد چه چیزی بازیابی شده، نه اینکه مدل واقعاً فقط بر همان evidence تکیه کرده است؛ ارزیابی groundedness لازم است.
