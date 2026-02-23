from sentence_transformers import SentenceTransformer
import chromadb
import json
import ollama
import re
import requests
from bs4 import BeautifulSoup
from sqlglot import select
import sqlglot
from sqlglot import exp
from datetime import datetime


embedding_model = SentenceTransformer("intfloat/e5-small-v2")

table_info_client = chromadb.PersistentClient(path="C:\last_project\make_domain_selector\\table_info")
table_info_collection = table_info_client.get_collection("table_info")

concept_client = chromadb.PersistentClient(path="C:\last_project\make_domain_selector\\concept_store")
concept_collection = concept_client.get_collection("concept_all")

syntax_client = chromadb.PersistentClient(path="C:\last_project\make_domain_selector\\syntax_info")
syntax_collection = syntax_client.get_collection("syntax_info")


def embed_passages(texts):
    texts = [f"passage: {t}" for t in texts]
    return embedding_model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

def embed_query(text):
    """검색 쿼리 임베딩용"""
    if isinstance(text, str):
        text = [text]
    texts = [f"query: {t}" for t in text]
    return embedding_model.encode(texts, show_progress_bar=True).tolist()


# 한국어 질문 번역 + 의료 용어 추출하기
def translate(korean_text):
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": """Extract medical terms and translate the sentence to English. 
Distinguish between medical abbreviations and full medical terms.
For medical terms, be smart about what to include:
- If it's a compound medical concept, keep the full phrase: "blood pressure", "heart rate", "respiratory rate"
- If it's a substance/chemical with a generic modifier, drop the modifier: "lactate levels" → "lactate", "glucose concentration" → "glucose", "sodium values" → "sodium"
- Generic modifiers to remove: levels, values, concentration, amount, measurement

SCHEMA-LEVEL CONCEPTS (DO NOT map to lab items):
- length of stay
- LOS
- mortality
- death
- died
- discharge
- admission
- ICU
- ICU ward
- care unit
- hospital stay
- in-hospital death
- patient count
- admission count

If these appear, DO NOT extract them as medical measurement terms.
Do NOT map them to labevents or itemid-based concepts.
They are structural database concepts.

Return JSON format: {
  "translation": "...", 
  "abbreviations": ["NE", "MAP", "Cr"], 
  "terms": ["blood pressure", "lactate", "heart rate"]
}
Medical abbreviations are SHORT uppercase forms (2-5 characters like NE, MAP, Cr, WBC).
Extract the most meaningful medical term form."""
            },
            {
                "role": "user",
                "content": korean_text
            }
        ],
        options={"temperature": 0},
        keep_alive="10m"
    )

    result = json.loads(response['message']['content'])
    return result['translation'], result.get('abbreviations', []), result.get('terms', [])

# 의료 약어의 경우 해당 사이트에서 찾아내기
def fetch_kmle_html(term: str) -> str:
    url = "https://www.kmle.co.kr/search.php"
    params = {
        "Search": term,
        "EbookTerminology": "YES",
        "DictAll": "YES",
        "DictAbbreviationAll": "YES",
        "DictDefAll": "YES",
        "DictNownuri": "YES",
        "DictWordNet": "YES"
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.encoding = r.apparent_encoding  # EUC-KR 대응
    return r.text

def is_name_like(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if ";" in s:                 # 'national emergency; ...' 같은 설명형은 제외
        return False
    if len(s) > 60:              # 너무 긴 건 제외
        return False
    # 영문자 비율이 너무 낮으면 제외
    letters = sum(c.isalpha() for c in s)
    if letters / max(len(s), 1) < 0.5:
        return False
    return True

def extract_names_from_abbr_tables(term: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    term_upper = term.upper()

    names = []
    for table in soup.find_all("table"):
        # 테이블 내 행(줄)에서:
        # [약어] [풀네임] 형태로 나오는 케이스를 잡음
        for tr in table.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue

            left = tds[0].get_text(" ", strip=True).upper()
            right = tds[1].get_text(" ", strip=True)

            if left == term_upper and is_name_like(right):
                names.append(right.strip())

    # 중복 제거(순서 유지)
    uniq = []
    seen = set()
    for n in names:
        key = re.sub(r"\s+", " ", n).strip().lower()
        if key not in seen:
            seen.add(key)
            uniq.append(re.sub(r"\s+", " ", n).strip())

    return uniq

def clean_name(s: str) -> str:
    s = re.sub(r"^[^A-Za-z]+", "", s)  # 앞쪽 숫자/특수기호 제거
    s = re.sub(r"[^A-Za-z\s\-]", "", s)  # 영어, 공백, 하이픈만 허용
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def return_clean_medical_words(medical_short_word):
    clean_medical_words = []

    for word in medical_short_word:
        html = fetch_kmle_html(word)
        names = extract_names_from_abbr_tables(word, html)[0]
        clean_medical_words.append(clean_name(names))

    return clean_medical_words

# 실제 찾은 의료 용어를 db와 매칭
def medical_word_db_mapping(clean_medical_words):

    medical_word_doc = []
    medical_word_meta = []

    for word in clean_medical_words:
        # 검색 시 수정
        results = concept_collection.query(
            query_embeddings=embed_query(word),  # "query:" prefix 사용
            n_results=1
        )

        word_info = results['metadatas'][0][0]

        medical_word_doc.append(f'''WORD : {word}. \n It means that "{results["documents"][0][0]}" \n This information is in {results["metadatas"][0][0]['table']}. \n In the {word_info['table']} table, {word_info['column']} {word_info['values']} represents {word_info['name']}.''')
        medical_word_meta.append(word_info)

    return medical_word_doc, medical_word_meta

# ChromaDB에서 관련 테이블 검색
def return_table_list(text_en, medical_word_doc):

    if medical_word_doc:
        medical_info_block = "\n".join(medical_word_doc)
        user_content = f"""
Query: {text_en}

Medical Term Information:
{medical_info_block}

Extract ALL table names mentioned above.
Return ONLY a valid JSON array of table names.
"""
    else:
        user_content = f"""
Query: {text_en}

No explicit medical term mapping is available.
Based only on the query intent, determine relevant tables.

Return ONLY a valid JSON array of table names.
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": """You are a database schema analyzer.
Return ONLY a valid JSON array of real table names.
Do NOT return example placeholders like table1 or table2.
Do NOT explain.
JSON only."""
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        options={"temperature": 0},
        keep_alive="10m"
    )

    json_match = re.search(r'\[.*\]', response['message']['content'], re.DOTALL)
    selected_tables = json.loads(json_match.group()) if json_match else []

    return selected_tables

def return_table_detail_info(selected_tables):
    for_query_table_info = ''

    for table_name in selected_tables:
        results = table_info_collection.query(
            query_embeddings=embed_query(table_name),
            n_results=1,
            where={"table_name": table_name}
        )

        for_query_table_info += f"""
    TABLE: {table_name}

    COLUMNS:
    {results['documents'][0][0][results['documents'][0][0].find('Key column descriptions'):]}
    ---
    """
        
    return for_query_table_info

def find_match_column(text_en, for_query_table_info):
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "Extract relevant columns. Return ONLY JSON: {\"inputevents\": [\"col1\", \"col2\"], \"chartevents\": [\"col3\"]}"
            },
            {
                "role": "user",
                "content": f"Query: {text_en}\nMetadata: {for_query_table_info}\n\nJSON only."
            }
        ],
        options={"temperature": 0},
        keep_alive="10m"
    )

    json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
    matched_columns = json.loads(json_match.group()) if json_match else {}
    return matched_columns

def detect_sql_intent(text_en: str):
    text = text_en.lower()

    aggregation_detected = any(word in text for word in [
        "average", "mean", "count", "number of",
        "sum", "total", "minimum", "lowest",
        "maximum", "highest"
    ])

    grouping_detected = any(word in text for word in [
        "by", "per", "each", "grouped by"
    ])

    date_detected = any(word in text for word in [
        "today", "yesterday", "last", "recent",
        "days", "weeks", "months", "year"
    ])

    icu_detected = "icu" in text or "careunit" in text

    mortality_detected = any(word in text for word in [
        "death", "mortality", "expire", "died"
    ])

    exclusion_detected = any(word in text for word in [
        "without", "exclude", "not have", "no history"
    ])

    # join 감지는 matched_columns 길이 기반으로 하는 게 더 정확
    join_detected = False  # 기본값, 아래에서 따로 설정 가능

    return {
        "aggregation": aggregation_detected,
        "group_by": grouping_detected,
        "date_filter": date_detected,
        "icu_context": icu_detected,
        "mortality_logic": mortality_detected,
        "exclusion_logic": exclusion_detected,
        "join_required": join_detected
    }

def detect_join_requirement(matched_columns):
    tables = set()

    for col in matched_columns:
        # 예: "patients.subject_id"
        table = col.split(".")[0]
        tables.add(table)

    return len(tables) > 1

# 🔥 컬럼 hallucination 강제 차단 블록
column_safety_block = """
========================
COLUMN VALIDATION RULE
========================

CRITICAL:

- You MUST use ONLY column names that appear in "Available Columns".
- You MUST NOT invent, modify, infer, or guess column names.
- If a required column is not in Available Columns, DO NOT create it.
- If unsure, choose from Available Columns only.
- Column names must match EXACTLY (case-sensitive).
- Any invented column name makes the output INVALID.

Before returning JSON:
- Verify every column in "select", "where", "group_by", "having", "order_by"
  exists in Available Columns.

If any column is not in Available Columns,
internally regenerate the JSON before returning.
"""

def return_json_for_sqlglot(text_en, matched_columns, medical_word_meta):
    intent = detect_sql_intent(text_en)
    intent["join_required"] = detect_join_requirement(matched_columns)

    rag_query_text = f"""
    USER QUESTION:
    {text_en}

    STRUCTURAL REQUIREMENTS:
    - aggregation: {intent['aggregation']}
    - group_by: {intent['group_by']}
    - join_required: {intent['join_required']}
    - date_filter: {intent['date_filter']}
    - icu_context: {intent['icu_context']}
    - mortality_logic: {intent['mortality_logic']}
    - exclusion_logic: {intent['exclusion_logic']}

    Oracle flat SQL enforcement.
    """

    results = syntax_collection.query(
        query_embeddings=embed_query(rag_query_text),
        n_results=10
    )

    retrieved_rules_text = "\n\n".join(results["documents"][0])
    current_date = datetime.now().strftime('%Y-%m-%d')

    # 🔥 JSON 구조 블록을 f-string 밖으로 분리
    required_json_structure = """
    ========================
    REQUIRED JSON STRUCTURE
    ========================

    You MUST return JSON in EXACTLY this structure:

    {
    "select": [string],
    "from": string,
    "joins": [string],
    "where": [string],
    "group_by": [string],
    "having": [string],
    "order_by": [string],
    "distinct": boolean
    }

    Rules:
    - "select" must contain column expressions ONLY.
    - NEVER include DISTINCT inside select items.
    - DISTINCT must be controlled ONLY by the "distinct" boolean field.
    - If no joins exist, return empty array [].
    - If no where conditions exist, return empty array [].
    - If no group_by needed, return empty array [].
    - Arrays must never be null.
    - Boolean fields must be true or false.
    """

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": f"""Generate SQL structure JSON for Oracle Database.

    CURRENT DATE: {current_date}

    You MUST follow ALL SQL GENERATION RULES below.
    Violation of rules makes the output INVALID.

    ========================
    SQL GENERATION RULES
    ========================
    {retrieved_rules_text}

    ========================
    STRUCTURAL INTENT FLAGS
    ========================
    aggregation_detected: {intent['aggregation']}
    group_by_detected: {intent['group_by']}
    join_required: {intent['join_required']}
    date_filter: {intent['date_filter']}
    icu_context: {intent['icu_context']}
    mortality_logic: {intent['mortality_logic']}
    exclusion_logic: {intent['exclusion_logic']}

    ========================
    MANDATORY LOGIC ENFORCEMENT
    ========================

    1. If aggregation_detected is TRUE AND group_by_detected is TRUE
    THEN "group_by" array MUST NOT be empty.

    2. If aggregation_detected is TRUE AND group_by_detected is FALSE
    THEN return single aggregated result with empty group_by.

    3. If aggregation_detected is FALSE
    THEN do NOT use AVG, COUNT, SUM, MIN, MAX.

    4. If group_by_detected is TRUE
    THEN GROUP BY must include all non-aggregated SELECT columns.

    5. If icu_context is TRUE
    THEN ICU unit MUST use icustays.first_careunit.

    6. If mortality_logic is TRUE
    THEN mortality indicator must follow mortality rules.

    7. If any mandatory rule is violated,
    internally regenerate before returning JSON.

    ========================
    GLOBAL RESTRICTIONS
    ========================
    - Flat query structure only.
    - FROM must contain exactly one base table.
    - No subqueries in FROM or JOIN.
    - No table aliases.
    - Always use full table names.
    - NEVER use SELECT *.
    - NEVER output raw boolean expressions.
    - Use CASE WHEN ... THEN 1 ELSE 0 END for indicators.
    - Use icustays.los for ICU length of stay.
    - Use HAVING only for aggregated filtering.
    - Use ORDER BY only if explicitly required.
    - Use TRUNC(SYSDATE) for "today".
    - DISTINCT must be false unless duplication risk exists.
    - Do not invent columns or tables.

    NUMERIC OUTPUT FORMATTING RULE:

    1. If AVG() is used, it MUST be wrapped with ROUND(expression, 2).
    2. If SUM() produces decimal values, wrap with ROUND(expression, 2).
    3. Do NOT round COUNT() results.
    4. Alias must reflect rounded meaning.

    Return ONLY valid JSON in the required format.
    Do not explain.

    {required_json_structure}

    {column_safety_block}

    """
            },
            {
                "role": "user",
                "content": f"""
    Query: {text_en}
    Available Columns: {matched_columns}
    Metadata Filters: {medical_word_meta}

    Remember:
    - Today is {current_date}
    - When user says "오늘" or "today", use TRUNC(SYSDATE) in WHERE clause
    - DO NOT use any table aliases
    - Use full table names

    Analyze the query intent and select appropriate columns.
    Do not include duplicate columns in SELECT.
    """
            }
        ],
        options={"temperature": 0}
    )

    json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
    sql_json = json.loads(json_match.group())

    # 🔧 distinct 안전 처리 (주석 수정)
    if 'distinct' not in sql_json:
        sql_json['distinct'] = False
    
    return sql_json
    # 현재 날짜 정보 추가

def return_sql(sql_json):

    query = select(*sql_json["select"])

    # DISTINCT
    if sql_json.get("distinct"):
        query = query.distinct()

    query = query.from_(sql_json["from"])

    # JOIN
    for join in sql_json.get("join", []):
        query = query.join(join["table"], on=join["on"])

    # WHERE
    for condition in sql_json.get("where", []):
        query = query.where(condition)

    # ✅ GROUP BY
    if sql_json.get("group_by"):
        query = query.group_by(*sql_json["group_by"])

    # ✅ HAVING
    for condition in sql_json.get("having", []):
        query = query.having(condition)

    # ✅ ORDER BY
    if sql_json.get("order_by"):
        query = query.order_by(*sql_json["order_by"])

    sql = query.sql(dialect="oracle")

    return sql.strip().replace("\n", " ").replace(";", "")

def change_bind_query(query):
    try:
        parsed = sqlglot.parse_one(query, dialect="oracle")

        bind_values = {}
        bind_idx = 1

        for literal in parsed.find_all(exp.Literal):
            parent = literal.parent
            
            if parent and "format" in parent.args and parent.args["format"] is literal:
                continue

            bind_name = f"v{bind_idx}"
            bind_idx += 1

            if literal.is_string:
                val = literal.this  # 실제 문자열 값

                # YYYY-MM 형태인지 체크 (안전)
                if (
                    isinstance(val, str)
                    and len(val) >= 7
                    and val[:4].isdigit()
                    and val[4] == "-"
                ):
                    year = int(val[:4])

                    if year % 1000 == 2:
                        new_year = year - 188
                        bind_values[bind_name] = f"{new_year}{val[4:]}"
                    else:
                        bind_values[bind_name] = val
                else:
                    bind_values[bind_name] = val

            elif literal.is_number:
                bind_values[bind_name] = float(literal.this)

            # ⭐ Literal → Parameter 로 교체
            literal.replace(exp.Parameter(this=bind_name))

        res = parsed.sql(dialect="oracle").replace("@",":")
        return res, bind_values
    except:
        return query, {}

def validate_sql_syntax(cur, sql_text):
    try:
        cur.execute(f'EXPLAIN PLAN FOR {sql_text}')
        cur.execute("SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY)")
        plan_text = "\n".join(r[0] for r in cur.fetchone())
        if len(plan_text) > 0:
            return True
    except:
        return False

def llm_answer(korean_text, cur):
    text_en, medical_short_word, medical_real_term = translate(korean_text)

    # None 방지
    medical_short_word = medical_short_word or []
    medical_real_term = medical_real_term or []

    # 1️⃣ 약어가 있을 때만 정제
    if len(medical_short_word) > 0:
        clean_medical_words = return_clean_medical_words(medical_short_word)
    else:
        clean_medical_words = []

    # 2️⃣ 매핑 대상 의료 단어 리스트 구성
    all_medical_terms = clean_medical_words + medical_real_term

    # 3️⃣ 의료 용어가 있을 때만 DB 매핑
    if len(all_medical_terms) > 0:
        medical_word_doc, medical_word_meta = medical_word_db_mapping(all_medical_terms)
    else:
        medical_word_doc = []
        medical_word_meta = []

    # 4️⃣ 테이블 선택 (medical_word_doc가 빈 리스트여도 동작하도록)
    selected_tables = return_table_list(text_en, medical_word_doc)

    for_query_table_info = return_table_detail_info(selected_tables)
    matched_columns = find_match_column(text_en, for_query_table_info)

    sql_json = return_json_for_sqlglot(text_en, matched_columns, medical_word_meta)
    sql = return_sql(sql_json)

    bind_query, bind_dict = change_bind_query(sql)

    ok = validate_sql_syntax(cur, sql)
    if ok:
        return sql, sql_json, bind_query, bind_dict

# 생성된 sql을 설명하는 함수 
def generate_medical_sql_explanation_json(
    question_text,
    sql_structure_json,
    execution_summary=None
):
    current_date = datetime.now().strftime('%Y-%m-%d')

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": f"""
You are an explanation-only model for medical SQL queries.

CRITICAL RULES:
- Output MUST be written entirely in Korean.
- Return ONLY valid JSON.
- Maximum 80 Korean words.
- Use simple, clinical-friendly language.
- DO NOT mention SQL functions (AVG, ROUND, GROUP BY, etc.).
- Describe logic conceptually (e.g., "평균 계산", "병동 기준으로 구분").
- No clinical interpretation.
- No inference beyond provided JSON.
- Do NOT generate SQL.

MANDATORY:
- If WHERE is null or empty → 반드시 "전체 데이터를 대상으로 계산"이라고 명시.
- If JOIN is null or empty → "단일 테이블 기반 계산"이라고 간단히 표현.

OUTPUT FORMAT:

{{
  "explanation": "..."
}}
"""
            },
            {
                "role": "user",
                "content": f"""
Clinician Question:
{question_text}

SQL Structure JSON:
{json.dumps(sql_structure_json, indent=2)}

Execution Summary:
{execution_summary if execution_summary else "None provided."}

Generate concise explanation JSON in Korean.
"""
            }
        ],
        options={"temperature": 0}
    )

    json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
    explanation_json = json.loads(json_match.group())

    return explanation_json['explanation']

# 실제 SQL 추출 시 사용된 의료 용어 찾기
def extract_used_medical_terms(sql_json, medical_word_meta):
    used_terms = []

    # WHERE + JOIN ON 전체 조건 문자열로 합치기
    conditions = []

    conditions.extend(sql_json.get("where", []))

    for join in sql_json.get("join", []):
        conditions.append(join.get("on", ""))

    full_condition_text = " ".join(conditions)

    for meta in medical_word_meta:
        table = meta.get("table")
        column = meta.get("column")
        values = meta.get("values", "")
        name = meta.get("name")

        # values가 문자열 "[220052, 225312]" 형태라면 정리
        value_list = re.findall(r"\d+", values)

        for v in value_list:
            pattern = rf"{table}\.{column}.*{v}"
            if re.search(pattern, full_condition_text):
                used_terms.append(name)
                break
    
    return used_terms

from sqlglot import parse_one
from sqlglot.expressions import Table

def extract_tables(sql):
    parsed = parse_one(sql)
    tables = {table.name for table in parsed.find_all(Table)}
    return list(tables)