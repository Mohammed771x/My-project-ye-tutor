from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI

# =========================
# إعداد التطبيق
# =========================



app = FastAPI(title="YE - Pro Student Tutor v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OPENROUTER_API_KEY = "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"
GROQ_API_KEY = "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"







openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key= OPENROUTER_API_KEY
)




groq_client = Groq(
    api_key= GROQ_API_KEY
)










# =========================
# متغيرات وموارد
# =========================
BASE_SUBJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "subjects")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# جلسات المستخدمين (لتخزين pending pages أو pending exams)
sessions: Dict[str, Dict[str, Any]] = {}

# حدود وسياسات
MAX_PAGES_EXPLAIN_SUMMARY = 3      # أقصى صفحات لشرح/تلخيص عند الإدخال
MAX_PAGES_EXAMS = 5                # أقصى صفحات لبحث الوزاري عبر الصفحات
EXAMS_BATCH_SIZE = 10              # دفعة عرض أسئلة وزاري
UNIT_BATCH_PAGES = 3               # عدد صفحات في كل دفعة عند شرح/تلخيص الوحدة
QA_TOP_K = 3                       # عدد نتائج FAISS للسؤال

# =========================
# موديل الطلب
# =========================
class AskRequest(BaseModel):
    user_id: str
    subject: str             # اسم المادة كما في المجلد (مثال: احياء)
    logic_type: int = 1      # 1 = صارم من الكتاب
    mode: str                # "شرح" أو "تلخيص" أو "سؤال" أو "وزاري"
    input_type: str          # "صفحة" أو "وحدة" أو "برومت"
    content: str             # نص المستخدم أو أرقام الصفحات أو "كمل"/"وقف" أو "علمني"
    summary_level: int = 3   # 1..5 (للتلخيص)
    unit_name: Optional[str] = None # الحقل الجديد لاستلام الوحدة من الفرونت إند
    lesson_name: Optional[str] = None # اسم الدرس (للرياضيات)
# =========================
# دوال مساعدة لقراءة الكتب والأسئلة
# =========================
def subject_book_path(subject: str) -> str:
    return os.path.join(BASE_SUBJECTS_DIR, subject, f"{subject}.json")

def subject_exams_dir(subject: str) -> str:
    return os.path.join(BASE_SUBJECTS_DIR, subject, "exams")

def load_json_safe(path: str):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def find_unit(book_data: List[dict], query: str):
    """البحث عن الوحدة بالاسم أو بالرقم (مطابقة جزئية)"""
    q = query.strip()
    for unit in book_data:
        if q == unit.get("اسم_الوحدة") or q == str(unit.get("رقم_الوحدة")):
            return unit
    # محاولة مطابقة جزئية
    for unit in book_data:
        if q in unit.get("اسم_الوحدة", ""):
            return unit
    return None

def fetch_pages_by_numbers(book_data: List[dict], page_nums: List[int]):
    found = []
    missing = []
    for p in page_nums:
        found_flag = False
        for unit in book_data:
            for page in unit.get("الصفحات", []):
                if page.get("رقم_الصفحة") == p:
                    found.append(page)
                    found_flag = True
                    break
            if found_flag:
                break
        if not found_flag:
            missing.append(p)
    return found, missing
def extract_relevant_book_texts(book_data, query, top_k=5):
    """
    تبحث في الكتاب أولاً (FAISS + كلمات)
    وترجع نصوص الصفحات المرتبطة فعلياً بالموضوع
    """
    texts, metas = extract_all_texts_and_metas(book_data)

    # بحث دلالي
    sem_results, _ = faiss_search(texts, query, top_k=top_k)

    # بحث مباشر بالكلمات
    keywords = re.findall(r'[\u0600-\u06FF]{3,}', query)
    direct_hits = []

    for txt in texts:
        if any(k in txt for k in keywords):
            direct_hits.append(txt)

    # دمج بدون تكرار
    final_texts = []
    for t in direct_hits + sem_results:
        if t not in final_texts:
            final_texts.append(t)

    return final_texts[:top_k]

def extract_all_texts_and_metas(book_data: List[dict]):
    texts = []
    metas = []
    for unit in book_data:
        for page in unit.get("الصفحات", []):
            texts.append(page.get("نص_الصفحة", ""))
            metas.append({
                "unit": unit.get("اسم_الوحدة"),
                "page": page.get("رقم_الصفحة")
            })
    return texts, metas

def faiss_search(texts: List[str], query: str, top_k: int = QA_TOP_K):
    if not texts:
        return [], []
    emb = embed_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=min(top_k, len(texts)))
    results = []
    idxs = []
    for i in I[0]:
        results.append(texts[i])
        idxs.append(i)
    return results, idxs

# =========================
# برومبتات جاهزة
# =========================
def system_prompt_strict_explain(subject: str):
    return (
       f"أنت الآن في وضع مدرس داخل الصف لمادة {subject}. "
        "تتعامل مع الطالب وكأنك تشرح له أثناء الحصة الدراسية.\n\n"
        "القواعد الأساسية (مهم الالتزام بها بدقة):\n"
        "1) مصدر الإجابة الوحيد هو الكتاب المعطى لك فقط، ولا يُسمح باستخدام أي معلومات من خارج الكتاب.\n"
        "2) لا تضف معرفة عامة، ولا أمثلة خارجية، ولا اجتهاد شخصي.\n"
        "3) جميع الإجابات يجب أن تكون إما نقلًا مباشرًا من نص الكتاب أو شرحًا مبسطًا لمعنى موجود صراحة في الكتاب.\n\n"
        "أسلوب الشرح:\n"
        "- الأسلوب تعليمي، مرتب، وكأنك داخل الصف.\n"
        "- يمكن تقسيم الشرح إلى نقاط أو خطوات عند الحاجة.\n\n"
        "في حال لم تجد إجابة داخل نص الكتاب:\n"
        "قل بوضوح: (لا توجد إجابة مباشرة لهذا الطلب في نص الكتاب)، ولا تحاول التخمين أو الإضافة."
    )

def system_prompt_strict_summary(subject: str, level: int):
    levels = {1: "مفصل جداً", 2: "شامل", 3: "متوسط", 4: "مختصر", 5: "مختصر جداً في نقاط"}
    return (
        f"أنت ملخّص ماهر لمادة {subject}. التزم بالنص المقدم فقط. لخص بمستوى: {levels.get(level,'متوسط')}. "
        "لا تضف معلومات خارج النص. التنسيق يكون واضحًا ونقاط عند الحاجة."
    )

def system_prompt_strict_qa(subject: str):
    return (
        f"أنت مدرس يجيب مباشرة من نص كتاب مادة {subject}. أجب بجملة أو جملتين مقتبستين أو مستخلصة من النص فقط. "
        "إن لم تجد الإجابة داخل النص قل: 'عذراً، هذه المعلومة غير متوفرة في الكتاب'."
    )

def system_prompt_strict_exams(subject: str):
    return (
        f"أنت مساعد للأمتحانات لشهادة الثانوية في مادة {subject}. استخرج الأسئلة المطابقة من ملفات الأسئلة وفق معايير المستخدم. "
        "لا تضف أسئلة أو تغير في نصوص الأسئلة، فقط اعرض النصوص كما هي مع ذكر السنة والجزء ونوع السؤال."
    )

# =========================
# تعليمات الاستخدام (علمني)
# =========================
USAGE_HELP = {
    "صفحة": (
        "أنت في وضع الصفحات. اكتب أرقام الصفحات مفصولة بفواصل.\n"
        f"مثال: 45 أو 45,47\n"
        f"ملاحظة: الحد الأقصى للصفحات هنا هو {MAX_PAGES_EXPLAIN_SUMMARY} صفحات للشرح/تلخيص."
    ),
    "وحدة": (
        "أنت في وضع الوحدة. اكتب اسم الوحدة أو رقمها كما هو مكتوب في محتوى الكتاب.\n"
        f"سيتم شرح {UNIT_BATCH_PAGES} صفحة في كل مرة. اكتب 'كمل' للاستمرار أو 'وقف' لإنهاء الجلسة."
    ),
    "برومت": (
        "أنت في وضع البرومت. اكتب موضوعًا أو سؤالاً نصياً. يمكنك تحديد وحدة معينة لتسريع البحث ودقته."
    ),
    "سؤال": (
        "أنت في وضع السؤال. اكتب سؤالاً نصياً مفهوماً. يفضل تحديد الوحدة المختصة بالسؤال لنتائج أدق."
    ),
    "وزاري": (
        "أنت في وضع الأسئلة الوزارية.\n"
        "الصيغ المقبولة:\n"
        "- بحث بالوحدة: <سنة>,<اسم الوحدة>  مثال: 2018,الغدد الصماء\n"
        f"- بحث بالصفحات: <سنة>,<صفحة1>,<صفحة2>  (الحد الأقصى للصفحات هنا {MAX_PAGES_EXAMS})\n"
        "- بحث بالبرومت: <سنة>,<موضوع>  مثال: 2019,التنفس\n"
        "يمكنك كتابة 'الكل' بدلاً من السنة للبحث عبر كل السنوات."
    )
}
def ai_extract_topics_from_pages(page_texts: List[str]):
    """
    يفهم نصوص الصفحات ويستخرج المواضيع الأساسية للدرس
    """
    context = "\n".join(page_texts)

    prompt = f"""
أنت مدرس خبير في تحليل المناهج.
اقرأ نص الدرس التالي واستخرج:

1. المواضيع الرئيسية
2. المفاهيم الأساسية
3. ما الذي يركز عليه الدرس فعلياً

أعد النتيجة كنقاط مختصرة وواضحة.

نص الدرس:
{context}
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "أنت محلل مناهج دقيق جداً."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ خطأ استخراج المواضيع:", e)
        return ""

# =========================
# دوال المساعدة للـ وزاري
# =========================
def parse_exams_input(content: str):
    parts = [p.strip() for p in content.split(",") if p.strip() != ""]
    if not parts:
        return None, None
    year = parts[0]
    rest = parts[1:]
    return year, rest

def restrict_book_to_unit(book_data: List[dict], unit_name: str):
    """
    تحصر الكتاب داخل وحدة واحدة فقط
    """
    for unit in book_data:
        if unit_name == unit.get("اسم_الوحدة") or unit_name == str(unit.get("رقم_الوحدة")):
            return [unit]
        if unit_name in unit.get("اسم_الوحدة", ""):
            return [unit]
    return None

def collect_exam_questions_by_years(subject: str, years: List[str]):
    exams_dir = subject_exams_dir(subject)
    found = []
    if not os.path.isdir(exams_dir):
        return found
    for f in os.listdir(exams_dir):
        if not f.lower().endswith(".json"):
            continue
        file_year = os.path.splitext(f)[0].strip()
        if "الكل" not in years and file_year not in [str(y) for y in years]:
            continue
        data = load_json_safe(os.path.join(exams_dir, f))
        if not data:
            continue
        for block in data:
            questions = block.get("الاسئلة", [])
            for q in questions:
                found.append({
                    "سنة": block.get("سنة_الامتحان", file_year),
                    "الجزء": block.get("الجزء", ""),
                    "النوع": block.get("نوع_السؤال", ""),
                    "النص": q
                })
    return found



def help_for_mode(mode: str, input_type: str = None) -> str:
    if mode == "شرح":
        if input_type == "صفحة": return USAGE_HELP["صفحة"]
        if input_type == "وحدة": return USAGE_HELP["وحدة"]
        return USAGE_HELP["برومت"]
    if mode == "تلخيص":
        return USAGE_HELP["برومت"] + "\nاختر درجة التلخيص من 1 إلى 5."
    if mode == "سؤال": return USAGE_HELP["سؤال"]
    if mode == "وزاري": return USAGE_HELP["وزاري"]
    return "استخدم التطبيق لشرح أو تلخيص أو سؤال أو أسئلة وزارية."

def pages_with_headers(pages):
    blocks = []
    for p in pages:
        blocks.append(f"📄 الصفحة {p['رقم_الصفحة']}:\n{p['نص_الصفحة']}")
    return "\n\n".join(blocks)

def enhanced_qa_search(book_data, query, top_k=5):
    texts, metas = extract_all_texts_and_metas(book_data)
    sem_results, idxs = faiss_search(texts, query, top_k=top_k)
    keywords = re.findall(r'[\u0600-\u06FF\w]{3,}', query)
    direct_hits = []
    direct_idxs = []
    for i, txt in enumerate(texts):
        if any(k in txt for k in keywords):
            direct_hits.append(txt)
            direct_idxs.append(i)
    final_texts = []
    final_idxs = []
    for t, i in zip(direct_hits, direct_idxs):
        if i not in final_idxs:
            final_texts.append(t)
            final_idxs.append(i)
    for t, i in zip(sem_results, idxs):
        if i not in final_idxs:
            final_texts.append(t)
            final_idxs.append(i)
    return final_texts[:top_k], final_idxs[:top_k]

def filter_exams_by_keyword(questions: list, keyword: str):
    """
    ترجع كل الأسئلة الوزارية التي تحتوي على الكلمة أو العبارة المطلوبة
    """
    keyword = keyword.strip()
    if not keyword:
        return []

    matched = []
    for q in questions:
        text = q.get("النص", "")
        if keyword in text:
            matched.append(q)

    return matched


def extract_keywords(text: str):
    text = normalize_arabic(text)
    words = re.findall(r'[\u0600-\u06FF]{3,}', text)
    return set(words)



def filter_and_rank_exams(questions: list, user_text: str):
    """
    - أي سؤال يحتوي على كلمة واحدة على الأقل يطلع
    - يتم ترتيب الأسئلة حسب عدد الكلمات المتطابقة (الأكثر أولاً)
    """
    user_keywords = extract_keywords(user_text)
    if not user_keywords:
        return []

    scored_questions = []

    for q in questions:
        q_text = normalize_arabic(q.get("النص", ""))
        score = 0

        for kw in user_keywords:
            if kw in q_text:
                score += 1

        if score > 0:
            scored_questions.append((score, q))

    # ترتيب: الأعلى تطابقاً أولاً
    scored_questions.sort(key=lambda x: x[0], reverse=True)

    return [q for score, q in scored_questions]

def normalize_arabic(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    # إزالة التشكيل
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)

    # توحيد الحروف
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # إزالة أل التعريف
    text = re.sub(r'\bال', '', text)

    # إزالة أي شيء غير حروف عربية
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    return text
def extract_keywords(query: str):
    normalized = normalize_arabic(query)
    words = normalized.split()

    # تجاهل الكلمات القصيرة جداً
    return [w for w in words if len(w) >= 3]
def filter_exams_smart(questions: list, query: str, min_hits: int = 2):
    """
    ترجع الأسئلة التي تطابق الموضوع بعدد كافٍ من الكلمات
    """
    keywords = extract_keywords(query)
    if not keywords:
        return []

    matched = []

    for q in questions:
        q_text = normalize_arabic(q.get("النص", ""))
        hits = 0

        for w in keywords:
            if w in q_text:
                hits += 1

        if hits >= min_hits:
            matched.append(q)

    return matched


# =========================
# دوال المساعدة للـ رياضيات
# =========================

def system_prompt_math_explain():
    return (
        "أنت مدرس رياضيات تشرح من ملخص الطالب فقط.\n"
        "القواعد:\n"
        "1) الشرح يكون بنفس أسلوب الملخص.\n"
        "2) لا تضف قوانين غير موجودة.\n"
        "3) الشرح يكون تدريجي وبسيط.\n"
        "4) عند الأمثلة: اشرح خطوة خطوة كما هي.\n"
         "5) أشرح باللغة العربية فقط.ذى"
    )


def load_math_lesson(branch: str, lesson_name: str):
    """
    branch: تفاضل / تكامل / هندسة / جبر
    lesson_name: اسم الدرس
    """
    base = os.path.join(BASE_SUBJECTS_DIR, "رياضيات", branch)
    if not os.path.isdir(base):
        return None

    for f in os.listdir(base):
        if not f.endswith(".json"):
            continue
        name = os.path.splitext(f)[0]
        if lesson_name in name:
            return load_json_safe(os.path.join(base, f))

    return None



def system_prompt_math_explain():
    return (
        "أنت مدرس رياضيات محترف.\n"
        "اشرح الدرس للطالب شرحًا تعليميًا واضحًا بأسلوب مبسط.\n"
        "التزم بالمعلومات والقوانين الموجودة في النص فقط.\n"
        "لا تنسخ النص حرفيًا، بل اشرح المعنى.\n"
        "استخدم أمثلة من النص إن وجدت.\n"
        "الشرح يكون متسلسل وكأنك تشرح لطالب داخل الصف.\n"
        "التزم بالشرح باللغة العربية , القوانين والتعريفات وكل شي باللغة العربية \n"
    )


def explain_math_lesson(lesson: dict):
    content = json.dumps(lesson, ensure_ascii=False, indent=2)

    prompt = f"""
هذا ملخص درس رياضيات:

{content}

المطلوب:
اشرح هذا الدرس شرحًا تعليميًا واضحًا للطالب.
لاتكتب اي كلمة انجليزية او كلمة غير عربية اثناء الشرح
الشرح باللغة: العربية الفصحى فقط (يمنع الصينية والإنجليزية)
2. الرموز: استخدم (جا، جتا، ظا) و (س، ص) والرموز العربية حصراً.\n
- اكتب الشرح باللغة العربية فقط
- اكتب الشرح باللغة العربية.
- لا تستخدم LaTeX
- لا تستخدم \text أو \frac
- اكتب المعادلات كنص عادي
مثال: ص = 2س² + 1

"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt_math_explain()},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
import re


def solve_math_question_from_lesson(lesson: dict, user_text: str):
    lesson_text = json.dumps(lesson, ensure_ascii=False, indent=2)
    
    # التعليمات الصارمة للغة العربية والرموز
    system_prompt = (
        "أنت مدرس رياضيات محترف في تطبيق 'مسار'. التزم بالقواعد التالية:\n"
        "1. اللغة: العربية الفصحى فقط (يمنع الصينية والإنجليزية).\n"
        "- عند استلام سؤال، ابحث أولاً في الأمثلة الموجودة في بيانات الدرس.""""
   - إذا كان السؤال مطابقاً لمثال موجود، اذكر الحل النموذجي له.
   - إذا كان السؤال مشابهاً لمثال (باختلاف الأرقام أو الرموز)، حل المسألة بنفس الخطوات والمنطق المتبع في ذلك المثال تماماً.
"""        "2. الرموز: استخدم (جا، جتا، ظا) و (س، ص) حصراً.\n"
"""       - اكتب الشرح باللغة العربية فقط
- اكتب الشرح باللغة العربية.
- لا تستخدم LaTeX
- لا تستخدم \text أو \frac
- اكتب المعادلات كنص عادي
مثال: ص = 2س² + 1
"""
  "3. وضح القوانين المستخدمة في الحل.\nنهاية التعليمات."
  
    )

    try:
        # الاتصال بـ Groq باستخدام موديل DeepSeek R1 المقطر
        response = openrouter_client.chat.completions.create(
            model="tngtech/deepseek-r1t-chimera:free", # الموديل الأفضل حالياً
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"بيانات الدرس:\n{lesson_text}\n\nسؤال الطالب: {user_text}"}
            ],
            temperature=0.2
        )

        full_answer = response.choices[0].message.content

        # تنظيف "خطوات التفكير" (Thinking)
        clean_answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL).strip()
        
        # تحويل يدوي لأي رموز إنجليزية قد تفلت من الموديل
        clean_answer = clean_answer.replace("sin", "جا").replace("cos", "جتا").replace("tan", "ظا")
        clean_answer = clean_answer.replace("x", "س").replace("y", "ص")
        
        return clean_answer
    except Exception as e:
        return f"عذراً، حدث خطأ في محرك السؤال: {str(e)}"

def handle_math_request(req: AskRequest):
    user_id = req.user_id
    mode = req.mode.strip()          # شرح / سؤال
    branch = req.unit_name           # تفاضل / تكامل / ...
    user_text = req.content.strip()
    
    # نعتمد على اسم الدرس المرسل في الحقل المخصص له
    lesson_name = req.lesson_name

    # ======================
    # علمني
    # ======================
    if user_text == "علمني":
        return {
            "answer": (
                "📘 أنت في قسم الرياضيات:\n\n"
                "🟢 شرح الدرس:\n"
                "- اختر الفرع\n"
                "- اختر الدرس\n"
                "- اضغط (شرح الدرس)\n"
                "- بعد الشرح يمكنك السؤال عن أي نقطة في الشرح\n\n"
                "🔵 سؤال:\n"
                "- اختر الدرس\n"
                "- اكتب المسألة\n"
                "- سيحلها الذكاء الاصطناعي بنفس أسلوب الملخص\n"
            ),
            "session_active": False
        }

    # ======================
    # تحقق من الفرع
    # ======================
    if not branch:
        return {"answer": "⚠️ اختر فرع الرياضيات أولاً."}

    # ======================
    # 🟢 قسم شرح الدرس
    # ======================
    if mode == "شرح":
        
        if not lesson_name:
             return {"answer": "⚠️ يجب اختيار الدرس أولاً."}

        sess = sessions.get(user_id)

        # نحدد هل هذا طلب شرح جديد (ضغط الزر) أم سؤال متابعة؟
        is_start_new_explanation = (user_text == lesson_name)

        # 🟢 1) بداية شرح درس جديد
        if is_start_new_explanation or not sess or sess.get("lesson_name") != lesson_name:
            
            if not is_start_new_explanation and (not sess or sess.get("lesson_name") != lesson_name):
                 return {"answer": "⚠️ الرجاء ضغط زر (شرح الدرس) لتأسيس الشرح أولاً."}

            sessions.pop(user_id, None)

            lesson = load_math_lesson(branch, lesson_name)
            if not lesson:
                return {"answer": f"❌ لم أجد درس '{lesson_name}' في {branch}."}

            explanation = explain_math_lesson(lesson)

            sessions[user_id] = {
                "subject": "رياضيات",
                "mode": "math_explain",
                "lesson": lesson,
                "lesson_name": lesson_name,
                "last_explanation": explanation
            }

            return {
                "answer": explanation,
                "session_active": False  # ✅ تعديل: إخفاء أزرار كمل/وقف في الرياضيات
            }

        # 🟢 2) سؤال داخل نفس الشرح (متابعة)
        if sess and sess.get("mode") == "math_explain":
            lesson = sess["lesson"]
            last_explanation = sess["last_explanation"]

            prompt = f"""
الشرح السابق:
{last_explanation}

سؤال الطالب:
{user_text}

المطلوب:
- أجب على السؤال بالاعتماد على نفس الدرس فقط.
لاتكتب اي كلمة انجليزية او كلمة غير عربية اثناء الشرح
- وضّح الفكرة بأسلوب تعليمي مرتبط بالشرح السابق.
- اكتب الشرح باللغة العربية فقط
- لا تستخدم LaTeX
- لا تستخدم \text أو \frac
- اكتب المعادلات كنص عادي
مثال: ص = 2س² + 1
"""

            response = openrouter_client.chat.completions.create(
                model="tngtech/deepseek-r1t-chimera:free",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt_math_explain()},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "session_active": False  # ✅ تعديل: إخفاء أزرار كمل/وقف مع استمرار الشات
            }

        return {
            "answer": "⚠️ اضغط (شرح الدرس) أولاً.",
            "session_active": False
        }

    # ======================
    # 🔵 قسم السؤال (منفصل تماماً)
    # ======================
    # ابحث عن هذا الجزء داخل handle_math_request
    if mode == "سؤال":
        # استخدام ملف افتراضي في حال لم يرسل التطبيق اسماً للدرس
        current_lesson = req.lesson_name or "مشتقة الدوال المثلثية الدائرية"
        
        lesson = load_math_lesson(branch, current_lesson)
        if not lesson:
            return {"answer": f"⚠️ لم أجد ملف الدرس: {current_lesson}"}

        answer = solve_math_question_from_lesson(lesson, user_text)
        return {"answer": answer, "session_active": False}

# =========================
# نقطة النهاية /ask
# =========================
@app.get("/math/lessons")
def get_math_lessons(branch: str):
    math_root = os.path.join(BASE_SUBJECTS_DIR, "رياضيات")
    branch_path = os.path.join(math_root, branch)

    if not os.path.isdir(branch_path):
        print("❌ branch folder not found")
        return []

    lessons = []

    for fname in os.listdir(branch_path):
        if not fname.endswith(".json"):
            continue

        full_path = os.path.join(branch_path, fname)
        data = load_json_safe(full_path)

        if not data:
            continue

        lesson_name = data.get("اسم_الدرس")
        if lesson_name:
            lessons.append(lesson_name)

    print("✅ lessons:", lessons)
    return lessons



@app.post("/ask")
async def ask(req: AskRequest):
    user_id = req.user_id
    # 🔁 تغيير المادة = شات جديد
    prev = sessions.get(req.user_id)
    if prev and prev.get("subject") != req.subject:
      sessions.pop(req.user_id, None)

    subject = req.subject.strip()
    mode = req.mode.strip()
    input_type = req.input_type.strip()
    content = req.content.strip()
    # ===============================
# 📐 مسار خاص لمادة الرياضيات
# ===============================
    if subject == "رياضيات":
        return handle_math_request(req)


    book_path = subject_book_path(subject)
    book_data = load_json_safe(book_path)
    if not book_data:
        return {"answer": f"المادة '{subject}' غير متوفرة."}

    control_lower = content.strip().lower()
    if control_lower in ["وقف", "خلاص", "شكرا", "إلغاء"]:
        sessions.pop(user_id, None)
        return {"answer": "تم إيقاف الجلسة الحالية.", "session_active": False}

    if content.strip() == "علمني":
        return {"answer": help_for_mode(mode, input_type)}
    
    # منطق الفلترة حسب الوحدة (لأطوار الشرح والتلخيص والسؤال بنوع برومت)
    target_data = book_data
    if input_type == "برومت" or mode == "سؤال":
        if req.unit_name and req.unit_name not in ["الكل", ""]:
            unit = find_unit(book_data, req.unit_name)
            if unit:
                target_data = [unit] # هنا يتم حصر البحث داخل صفحات الوحدة المختارة فقط
            else:
                return {"answer": f"عذراً، لم أجد الوحدة '{req.unit_name}' في الكتاب."}

    # "كمل" للمتابعة
    

    # 1) وضع "شرح"
    if mode == "شرح":
        if input_type == "صفحة":
            numbers = re.findall(r'\d+', content)
            if not numbers: return {"answer": "صيغة غير صحيحة."}
            page_nums = [int(n) for n in numbers]
            if len(page_nums) > MAX_PAGES_EXPLAIN_SUMMARY: return {"answer": f"الحد {MAX_PAGES_EXPLAIN_SUMMARY}."}
            found_pages, missing = fetch_pages_by_numbers(book_data, page_nums)
            if not found_pages: return {"answer": "الصفحات غير موجودة."}
            context_text = pages_with_headers(found_pages)
            system_prompt = system_prompt_strict_explain(subject)
            try:
                response = openrouter_client.chat.completions.create(model="nvidia/nemotron-3-nano-30b-a3b:free", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"نص الكتاب:\n{context_text}\n\nطلب الطالب: اشرح المحتوى أعلاه."}], temperature=0.1)
                answer = response.choices[0].message.content
            except Exception: answer = "خطأ في التوليد."
            if missing: answer += f"\n\nملاحظة: لم نجد {missing}."
            return {"answer": answer, "references": [f"ص {p.get('رقم_الصفحة')}" for p in found_pages], "session_active": False}

        

        if input_type == "برومت":
            results, idxs = enhanced_qa_search(target_data, content, top_k=5)
            if not results: return {"answer": "لم أجد مقاطع صلة."}
            context_text = "\n".join(results)
            system_prompt = system_prompt_strict_explain(subject)
            try:
                response = openrouter_client.chat.completions.create(
                    model="nvidia/nemotron-3-nano-30b-a3b:free", 
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": f"نص الكتاب:\n{context_text}\n\nطلب الطالب: اشرح الموضوع '{content}' بناءً على النص أعلاه."}
                    ], 
                    temperature=0.1
                )
                answer = response.choices[0].message.content
            except Exception: answer = "خطأ في التوليد."
            _, metas = extract_all_texts_and_metas(target_data)
            refs = [f"{metas[i]['unit']} - ص {metas[i]['page']}" for i in idxs]
            return {"answer": answer, "references": refs, "session_active": False}

    # 2) وضع "تلخيص"
    if mode == "تلخيص":
        if input_type == "صفحة":
            numbers = re.findall(r'\d+', content)
            page_nums = [int(n) for n in numbers]
            found_pages, missing = fetch_pages_by_numbers(book_data, page_nums)
            context_text = pages_with_headers(found_pages)
            system_prompt = system_prompt_strict_summary(subject, req.summary_level)
            try:
                response = openrouter_client.chat.completions.create(model="nvidia/nemotron-3-nano-30b-a3b:free", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"نص الكتاب:\n{context_text}\n\nالمطلوب: لخص المحتوى."}], temperature=0.15)
                answer = response.choices[0].message.content
            except Exception: answer = "خطأ في التلخيص."
            return {"answer": answer, "references": [f"ص {p.get('رقم_الصفحة')}" for p in found_pages], "session_active": False}

        if input_type == "برومت":
            texts, metas = extract_all_texts_and_metas(target_data)
            results, idxs = faiss_search(texts, content, top_k=QA_TOP_K)
            if not results: return {"answer": "لا توجد مقاطع."}
            context_text = "\n".join(results)
            system_prompt = system_prompt_strict_summary(subject, req.summary_level)
            try:
                response = openrouter_client.chat.completions.create(
                    model="nvidia/nemotron-3-nano-30b-a3b:free", 
                    messages=[
                        {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": f"نص الكتاب:\n{context_text}\n\nالمطلوب: لخص الموضوع '{content}'."}
                    ], 
                    temperature=0.15
                )
                answer = response.choices[0].message.content
            except Exception: answer = "خطأ."
            refs = [f"{metas[i]['unit']} - ص {metas[i]['page']}" for i in idxs]
            return {"answer": answer, "references": refs, "session_active": False}
    
    # 3) وضع "سؤال"
    # 3) وضع "سؤال" (تم تحسينه ليصبح ذكياً مثل الشرح لكن بإجابة مختصرة)
    if mode == "سؤال":
        # نستخدم enhanced_qa_search بدلاً من faiss_search
        # هذا يدمج البحث بالكلمات + البحث بالمعنى (نفس قوة الشرح)
        results, idxs = enhanced_qa_search(target_data, content, top_k=5)
        
        if not results: 
            return {"answer": "عذراً، لم أجد إجابة دقيقة في الكتاب."}
            
        context_text = "\n".join(results)
        
        # برومت مخصص: ذكي في الفهم، لكن مختصر في الرد
        system_prompt = (
            f"أنت مدرس لمادة {subject}. لديك نصوص من الكتاب المدرسي بالأسفل.\n"
            "المطلوب: أجب على سؤال الطالب إجابة دقيقة ومباشرة بناءً على النصوص.\n"
            "الشرط: الإجابة يجب أن تكون مختصرة (سطرين إلى ثلاثة كحد أقصى). أعط الزبدة فقط."
        )
        
        try:
            response = openrouter_client.chat.completions.create(
                model="nvidia/nemotron-3-nano-30b-a3b:free", 
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": f"نص الكتاب:\n{context_text}\n\nالسؤال: {content}"}
                ], 
                temperature=0.1
            )
            answer = response.choices[0].message.content
        except Exception: 
            answer = "حدث خطأ أثناء صياغة الإجابة."
            
        # استخراج المراجع
        _, metas = extract_all_texts_and_metas(target_data)
        # نحتاج التأكد من أن idxs صالحة (لأن enhanced قد يرجع اندكسات مكررة او مرتبة)
        refs = []
        for i in idxs:
            if i < len(metas):
                ref_str = f"{metas[i]['unit']} - ص {metas[i]['page']}"
                if ref_str not in refs:
                    refs.append(ref_str)

        return {"answer": answer, "references": refs, "session_active": False}
    # ===== متابعة الوزاري =====
    if content.strip() in ["كمل", "متابعة", "استمر"]:
        sess = sessions.get(user_id)

        if sess and sess.get("mode") == "وزاري" and sess.get("pending_exams"):
            pending = sess["pending_exams"]
            batch = pending[:EXAMS_BATCH_SIZE]
            sess["pending_exams"] = pending[EXAMS_BATCH_SIZE:]

            text = ""
            for i, q in enumerate(batch, 1):
              text += f"{i}. {q['النص']} (سنة {q['سنة']})\n"

            if sess["pending_exams"]:
               text += f"\n💡 المتبقي: {len(sess['pending_exams'])} سؤال."
            else:
                text += "\n✅ انتهت جميع الأسئلة."
                sessions.pop(user_id, None)

            return {
            "answer": text,
            "session_active": bool(sess.get("pending_exams"))
                 }
   
    # 4) وضع "وزاري"
    if mode == "وزاري":
        year_token, rest = parse_exams_input(content)

        if not year_token or not rest:
            return {
                "answer": "اكتب الصيغة هكذا:\nمثال: 2019,الدرقية"
            }

        years = [year_token] if year_token != "الكل" else ["الكل"]

        all_questions = collect_exam_questions_by_years(subject, years)
        if not all_questions:
            return {"answer": "لا توجد أسئلة لهذه السنة."}

        # الكلمة أو الموضوع
        keyword = ",".join(rest).strip()

        matched = filter_and_rank_exams(all_questions, keyword)


        if not matched:
            return {"answer": f"لم أجد أسئلة تحتوي على '{keyword}'."}

        # ==== العرض (10 + كمل) ====
        total = len(matched)
        first_batch = matched[:EXAMS_BATCH_SIZE]
        remaining = matched[EXAMS_BATCH_SIZE:]

        if remaining:
            sessions[user_id] = {
                "pending_exams": remaining,
                "mode": "وزاري",
                "subject": subject  # ✅ هذا هو التصحيح الضروري
            }

        text = f"✅ وجدت {total} سؤالاً يحتوي على '{keyword}':\n\n"
        for i, q in enumerate(first_batch, 1):
            text += f"{i}. {q.get('النص')} (سنة {q.get('سنة')})\n"

        if remaining:
            text += f"\n💡 تبقى {len(remaining)} سؤالاً. اكتب 'كمل' للمتابعة."

        return {
            "answer": text,
            "session_active": bool(remaining)
        }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
    
    
