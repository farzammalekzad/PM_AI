"""Deterministic synthetic Persian retrieval corpus and relevance labels."""
import json


THEMES = (
    'schedule', 'delay', 'minutes', 'contract', 'progress', 'procedure',
    'procurement', 'quality', 'communication', 'budget', 'safety',
)


def vector_for(theme):
    """Return a unit vector used only by the mocked evaluation embedder."""
    return [1.0 if item == theme else 0.0 for item in THEMES]


def document(document_id, chunk_no, name, content, theme):
    return {
        'document_id': document_id,
        'chunk_no': chunk_no,
        'name': name,
        'content': content,
        'embed_model': 'synthetic-persian-evaluation',
        'vector': json.dumps(vector_for(theme)),
    }


SYNTHETIC_DOCUMENTS = [
    document(1, 1, 'project-plan.txt', 'برنامه زمان‌بندی پروژه شامل نقاط عطف، مسئول هر فعالیت و تاریخ پایان است.', 'schedule'),
    document(1, 2, 'project-plan.txt', 'تاخیر بیش از سه روز باید ثبت و برای مدیر پروژه گزارش شود.', 'delay'),
    document(2, 1, 'meeting-minutes.txt', 'صورتجلسه تصمیم‌ها، مسئول اقدام‌ها و مهلت هر اقدام را ثبت می‌کند.', 'minutes'),
    document(2, 2, 'meeting-minutes.txt', 'حاضرین جلسه و زمان جلسه در ابتدای صورتجلسه نوشته می‌شوند.', 'minutes'),
    document(3, 1, 'contracts.txt', 'تغییر دامنه قرارداد فقط با درخواست کتبی و تایید طرفین معتبر است.', 'contract'),
    document(3, 2, 'contracts.txt', 'فسخ قرارداد نیازمند اعلام کتبی با مهلت مندرج در قرارداد است.', 'contract'),
    document(4, 1, 'progress-report.txt', 'گزارش پیشرفت هفتگی درصد پیشرفت، کارهای انجام‌شده و موانع را دارد.', 'progress'),
    document(4, 2, 'risk-register.txt', 'ثبت ریسک شامل احتمال، اثر، مالک ریسک و اقدام پاسخ است.', 'progress'),
    document(5, 1, 'procedures.txt', 'درخواست مرخصی باید در سامانه ثبت و توسط مدیر مستقیم تایید شود.', 'procedure'),
    document(5, 2, 'procurement.txt', 'خرید بالاتر از سقف مصوب به سه استعلام و تایید واحد تدارکات نیاز دارد.', 'procurement'),
    document(6, 1, 'quality.txt', 'تحویل کار پس از آزمون پذیرش و تایید معیارهای کیفیت انجام می‌شود.', 'quality'),
    document(6, 2, 'communication.txt', 'گزارش تغییرات مهم باید به ذی‌نفعان پروژه ارسال شود.', 'communication'),
    document(7, 1, 'budget.txt', 'انحراف بودجه ماهانه با بودجه مصوب مقایسه و در گزارش مالی ثبت می‌شود.', 'budget'),
    document(7, 2, 'safety.txt', 'هر حادثه ایمنی باید فوری ثبت و به مسئول ایمنی اطلاع داده شود.', 'safety'),
]


def case(question, relevant, theme=None, category=''):
    return {'question': question, 'relevant_ids': relevant, 'theme': theme, 'category': category}


SYNTHETIC_CASES = [
    case('نقاط عطف برنامه زمان‌بندی پروژه کجا ثبت می‌شوند؟', ['D1-C1'], 'schedule', 'project_management'),
    case('برای برنامه‌ریزی فعالیت‌ها چه اطلاعاتی لازم است؟', ['D1-C1'], 'schedule', 'project_management'),
    case('برنامه ريزي پروژه و تاريخ پايان فعاليت‌ها چیست؟', ['D1-C1'], 'schedule', 'persian_arabic_variant'),
    case('تاخیر پروژه از چه زمانی باید گزارش شود؟', ['D1-C2'], 'delay', 'project_delays'),
    case('اگر کار سه روز عقب افتاد چه اقدامی لازم است؟', ['D1-C2'], 'delay', 'semantic_paraphrase'),
    case('چه کسی از دیرکرد فعالیت‌ها مطلع می‌شود؟', ['D1-C2'], 'delay', 'project_delays'),
    case('صورتجلسه باید چه تصمیم‌هایی را ثبت کند؟', ['D2-C1'], 'minutes', 'meeting_minutes'),
    case('مسئول اقدام‌های جلسه در کدام سند مشخص می‌شود؟', ['D2-C1'], 'minutes', 'meeting_minutes'),
    case('چه کسانی در ابتدای صورتجلسه نوشته می‌شوند؟', ['D2-C2'], 'minutes', 'meeting_minutes'),
    case('تغییر دامنه قرارداد چگونه معتبر می‌شود؟', ['D3-C1'], 'contract', 'contracts'),
    case('اصلاح محدوده قرارداد به چه تاییدی نیاز دارد؟', ['D3-C1'], 'contract', 'semantic_paraphrase'),
    case('برای فسخ قرارداد چه نوع اعلانی لازم است؟', ['D3-C2'], 'contract', 'contracts'),
    case('گزارش پیشرفت هفتگی چه بخش‌هایی دارد؟', ['D4-C1'], 'progress', 'progress_reporting'),
    case('موانع پروژه را کجا گزارش می‌کنیم؟', ['D4-C1'], 'progress', 'progress_reporting'),
    case('ثبت ریسک شامل چه مشخصاتی است؟', ['D4-C2'], 'progress', 'organizational_procedures'),
    case('درخواست مرخصی چگونه تایید می‌شود؟', ['D5-C1'], 'procedure', 'organizational_procedures'),
    case('تاييد مرخصي توسط چه کسی انجام می‌شود؟', ['D5-C1'], 'procedure', 'persian_arabic_variant'),
    case('برای خرید بزرگ چند استعلام لازم است؟', ['D5-C2'], 'procurement', 'organizational_procedures'),
    case('واحد تدارکات چه خریدی را تایید می‌کند؟', ['D5-C2'], 'procurement', 'procurement'),
    case('تحویل کار پس از چه آزمونی انجام می‌شود؟', ['D6-C1'], 'quality', 'project_management'),
    case('پذیرش محصول به چه معیارهایی وابسته است؟', ['D6-C1'], 'quality', 'semantic_paraphrase'),
    case('تغییرات مهم پروژه به چه کسانی ارسال می‌شود؟', ['D6-C2'], 'communication', 'organizational_procedures'),
    case('اطلاع‌رسانی تغییرات پروژه برای ذی‌نفعان چگونه است؟', ['D6-C2'], 'communication', 'semantic_paraphrase'),
    case('انحراف بودجه ماهانه چگونه بررسی می‌شود؟', ['D7-C1'], 'budget', 'progress_reporting'),
    case('گزارش مالی چه مقایسه‌ای با بودجه دارد؟', ['D7-C1'], 'budget', 'budget'),
    case('حادثه ایمنی را چه زمانی باید ثبت کرد؟', ['D7-C2'], 'safety', 'organizational_procedures'),
    case('به چه کسی باید رخداد ایمنی را اطلاع داد؟', ['D7-C2'], 'safety', 'safety'),
    case('سیاست شرکت درباره غذای کارکنان چیست؟', [], None, 'unanswerable'),
    case('رمز عبور سامانه مالی چیست؟', [], None, 'unanswerable'),
    case('دفتر شرکت در کدام شهر است؟', [], None, 'unanswerable'),
    case('مدل تلفن مدیرعامل چیست؟', [], None, 'unanswerable'),
]


def synthetic_embedder(question, model):
    """Mocked local embedder for reproducible semantic and hybrid evaluation."""
    for item in SYNTHETIC_CASES:
        if item['question'] == question:
            return vector_for(item['theme']) if item['theme'] else [0.0] * len(THEMES)
    raise ValueError('synthetic evaluation question was not found')
