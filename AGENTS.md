# راهنمای مشارکت‌کنندگان

## قواعد قطعی

- برنامه آفلاین بماند؛ میزبان پیش‌فرض فقط `127.0.0.1` است.
- Gemma/Ollama، RTL فارسی، و دادهٔ SQLite فعلی را حفظ کنید.
- هرگز `ollama_company_data/`، فایل‌های SQLite، WAL یا SHM را حذف، reset یا commit نکنید.
- وابستگی، Docker، API خارجی، مدل embedding یا اتصال SQL Server را بدون تصمیم صریح اضافه نکنید.
- SQL فقط پارامتری باشد. SQL Server آینده فقط خواندنی، با حساب کم‌دسترسی و allowlist پرس‌وجو است.

## روند تغییر

1. ابتدا وضعیت Git و migrationهای موجود را بررسی کنید.
2. تغییر را کوچک و قابل بازبینی نگه دارید؛ برای تغییر schema migration افزایشی و قابل‌تکرار بنویسید، نه تغییر مخرب.
3. برای رفتار جدید، آزمون synthetic اضافه کنید؛ هرگز از دادهٔ سازمانی در آزمون استفاده نکنید.
4. اجرا کنید: `python -m unittest discover -s tests -v` و `python -m py_compile ollama_persian_rag.py`.
5. تغییرهای مستقل را با commitهای کوچک ثبت کنید. پیش از commit، `git diff --check` و `git status --short` را بررسی کنید.

## نکات فنی

- در هر connection SQLite، `foreign_keys=ON` و timeout فعال باشد.
- ورودی شبکه، base64، ZIP/OOXML، اندازهٔ متن و خطاهای Ollama را محدود و اعتبارسنجی کنید.
- متن UI با `textContent` درج شود؛ از `innerHTML` برای دادهٔ کاربر استفاده نکنید.
- برای فارسی، `dir="rtl"` در صفحه و `unicode-bidi: plaintext` در پیام‌ها حفظ شود؛ متن یا شناسه‌های لاتین باید امکان `dir=auto` داشته باشند.
