"""Small synthetic Persian answer-quality set; it contains no company content."""


ANSWER_CASES = [
    {
        'id': 'answerable_delay',
        'question': 'تاخیر پروژه از چه زمانی باید گزارش شود؟',
        'retrieved_ids': ['D1-C2'],
        'expected_abstention': False,
        'required_persian_phrases': ['بیش از سه روز'],
    },
    {
        'id': 'paraphrase_contract',
        'question': 'اصلاح محدوده قرارداد به چه تاییدی نیاز دارد؟',
        'retrieved_ids': ['D3-C1'],
        'expected_abstention': False,
        'required_persian_phrases': ['درخواست کتبی', 'تایید طرفین'],
    },
    {
        'id': 'partial_evidence',
        'question': 'آیا گزارش پیشرفت هفتگی علت همه موانع را مشخص می‌کند؟',
        'retrieved_ids': ['D4-C1'],
        'expected_abstention': False,
        'required_persian_phrases': ['موانع'],
    },
    {
        'id': 'unanswerable_password',
        'question': 'رمز عبور سامانه مالی چیست؟',
        'retrieved_ids': [],
        'expected_abstention': True,
        'required_persian_phrases': ['اطلاعات کافی'],
    },
]


MOCKED_ANSWERS = {
    'answerable_delay': 'تاخیر بیش از سه روز باید ثبت و برای مدیر پروژه گزارش شود. [D1-C2]',
    'paraphrase_contract': 'اصلاح محدوده قرارداد با درخواست کتبی و تایید طرفین معتبر است. [D3-C1]',
    'partial_evidence': 'منبع فقط وجود موانع در گزارش پیشرفت هفتگی را بیان می‌کند و علت همه آن‌ها را مشخص نمی‌کند. [D4-C1]',
    'unanswerable_password': 'اطلاعات کافی برای پاسخ به این سؤال در منابع بازیابی‌شده وجود ندارد.',
}
