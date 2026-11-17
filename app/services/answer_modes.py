"""Answer-mode validation and explicit grounded-response safeguards."""

GROUNDED = 'grounded'
HYBRID = 'hybrid'
GENERAL = 'general'
ANSWER_MODES = frozenset((GROUNDED, HYBRID, GENERAL))

GROUNDED_NO_EVIDENCE = (
    'برای پاسخ مستند، مدرک بازیابی‌شدهٔ کافی در اسناد شرکت وجود ندارد. '
    'لطفاً سند مرتبط را ثبت کنید یا پرسش را دقیق‌تر مطرح کنید.'
)


def normalize_answer_mode(value):
    """Validate a mode; callers choose the compatibility default when it is absent."""
    mode = str(value or GROUNDED).strip().lower()
    if mode not in ANSWER_MODES:
        raise ValueError('حالت پاسخ نامعتبر است')
    return mode


def is_simple_social_turn(question):
    """Allow only short, exact greetings to bypass document abstention."""
    normalized = str(question).strip().rstrip('.!؟? ').replace('،', '، ')
    return normalized in frozenset((
        'سلام', 'درود', 'سلام، خوبی', 'سلام، خوبید', 'سلام خوبی', 'سلام خوبید',
        'hello', 'hi',
    ))
