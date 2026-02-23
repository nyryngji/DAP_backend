import os
import re
import json
import requests
from bs4 import BeautifulSoup
import sqlglot
from sqlglot import exp
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from sqlglot import select, parse_one
from dotenv import load_dotenv
import oracledb

# 0. Setting
embedding_model = SentenceTransformer("intfloat/e5-small-v2")
client_table = chromadb.PersistentClient(path="C:\last_project\make_domain_selector\\table_info")
client_concept = chromadb.PersistentClient(path="C:\last_project\make_domain_selector\\concept_store")

FORBIDDEN = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b",
    r"\bDROP\b", r"\bALTER\b", r"\bTRUNCATE\b",
    r"\bCREATE\b", r"\bGRANT\b", r"\bREVOKE\b"
]

# DB
load_dotenv('.env')
oracledb.init_oracle_client(lib_dir=r"C:\instant_client\instantclient_21_19")


def embed_passages(texts):
    texts = [f"passage: {t}" for t in texts]
    return embedding_model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

def embed_query(text):
    """검색 쿼리 임베딩용"""
    if isinstance(text, str):
        text = [text]
    texts = [f"query: {t}" for t in text]
    return embedding_model.encode(texts, show_progress_bar=True).tolist()

# 1. 번역
def translate(korean_text: str) -> dict:
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "Extract medical terms and translate the sentence to English. Return JSON format: {\"translation\": \"...\", \"terms\": [\"term1\", \"term2\"]}. Keep medical abbreviations as-is (NE, MAP, Cr)."
            },
            {
                "role": "user",
                "content": korean_text
            }
        ],
        options={"temperature": 0.1}
    )

    result = json.loads(response['message']['content'])
    return result

# 2. 의료 용어 사전에서 itemid 찾기
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

def kmle_search(result: dict) -> list[str]:
    clean_medical_words = []

    for word in result['terms']:
        html = fetch_kmle_html(word)
        names = extract_names_from_abbr_tables(word, html)[0]
        clean_medical_words.append(clean_name(names))
    
    return clean_medical_words

# 3. Chromadb에서 찾은 단어 찾기
def concept_retriever(clean_medical_words: list[str]) -> tuple[list[str], list[dict]]:
    collection = client_concept.get_collection("concept_all")

    medical_word_doc = []
    medical_word_meta = []

    for word in clean_medical_words:
        results = collection.query(
            query_embeddings=embed_query(word),
            n_results=1
        )

        word_info = results['metadatas'][0][0]

        medical_word_doc.append(f'''WORD : {word}. \n It means that "{results["documents"][0][0]}" \n This information is in {results["metadatas"][0][0]['table']}. \n In the {word_info['table']} table, {word_info['column']} {word_info['values']} represents {word_info['name']}.''')
        medical_word_meta.append(word_info)

    return (medical_word_doc, medical_word_meta)

def table_retriever(result: dict, medical_word_doc: list[str]) -> str:
    collection = client_table.get_collection("table_info")

    query_text = ' '.join([result['translation']] + medical_word_doc)

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "Analyze the query and medical term information to identify ALL relevant tables. Return as JSON array of table names: [\"table1\", \"table2\"]. IMPORTANT: If multiple tables are mentioned in the medical term descriptions, return ALL of them."
            },
            {
                "role": "user",
                "content": f"""Query: {result['translation']}

    Medical Term Information:
    {query_text}

    Extract ALL table names mentioned in the medical term descriptions. Return them as a JSON array."""
            }
        ],
        options={"temperature": 0.1}
    )

    selected_tables = json.loads(response['message']['content'])
    for_query_table_info = ''

    for table_name in selected_tables:
        results = collection.query(
            query_embeddings=embed_query(table_name),
            n_results=1,
            where={"table_name": table_name}
        )

        table_schema = results['documents'][0][0]

        for_query_table_info += f"""
    TABLE: {table_name}

    COLUMNS:
    {results['documents'][0][0][results['documents'][0][0].find('Key column descriptions'):]}
    ---
    """
        
    return for_query_table_info

def column_selector(result: dict, for_query_table_info: str) -> dict:

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "Extract relevant columns. Return ONLY JSON: {\"inputevents\": [\"col1\", \"col2\"], \"chartevents\": [\"col3\"]}"
            },
            {
                "role": "user",
                "content": f"Query: {result['translation']}\nMetadata: {for_query_table_info}\n\nJSON only."
            }
        ],
        options={"temperature": 0}
    )

    json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
    columns = json.loads(json_match.group()) if json_match else {}

    return columns

# 6. sqlglot을 위한 json 파일 작성
def sql_json_builder(result: dict, columns: dict, medical_word_meta: list[str]) -> dict:
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": """Generate SQL structure JSON based on query intent. 

    CRITICAL RULES:
    1. Understand the query intent and select relevant columns that answer the question
    2. If table aliases are used in FROM/JOIN, ALL columns in SELECT must use those aliases (e.g., t1.col1, t2.col2)
    3. Use available columns strategically based on what the query is asking for
    4. Select columns that make results easy to read and understand for end users
    5. Include descriptive columns (names, labels, timestamps) when relevant for context
    6. NEVER include duplicate column names in SELECT - each column must appear only once
    8. Apply itemid filters from metadata

    Return ONLY valid JSON in this exact format:
    {
    "select": ["t1.col1", "t2.col2"],
    "from": "table1 AS t1",
    "join": [{"table": "table2 AS t2", "on": "t1.subject_id = t2.subject_id"}],
    "where": ["t1.itemid = 'value1'", "t2.itemid = 'value2'"]
    }"""
            },
            {
                "role": "user",
                "content": f"""
    Query: {result['translation']}
    Available Columns: {columns}
    Metadata Filters: {medical_word_meta}

    Analyze the query intent and select appropriate columns that answer the question.
    Ensure the result set is user-friendly and easy to interpret.
    Do not include any duplicate columns in the SELECT clause."""
            }
        ],
        options={"temperature": 0}
    )

    json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
    sql_json = json.loads(json_match.group())

    return sql_json

#. 7. SQL 쿼리 생성
def sql_renderer(sql_json: dict) -> str:
    query = select(*sql_json["select"]).from_(sql_json["from"])

    for join in sql_json["join"]:
        query = query.join(join["table"], on=join["on"])

    for condition in sql_json["where"]:
        query = query.where(condition)

    sql = query.sql(dialect='oracle')
    return sql

# 8. SQL 쿼리 조건 & 로직 요약
def summary_json(question: str, query_sql: str, sql_json: dict, medical_word_meta: list[dict]) -> dict:

    # 🔹 최소 FACT (속도 + 정확성 균형)
    fact = {
        "question": question,
        "where": sql_json.get("where", []),
        "concepts": [m.get("name") for m in (medical_word_meta or [])]
    }

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": """
Summarize the SQL logic into EXACTLY 3 short Korean bullet points.

Return ONLY JSON:

{
  "question": "...",
  "query": "...",
  "bullets": ["...", "...", "..."]
}

Rules:
- bullets must be concise but slightly descriptive (15~30 Korean characters).
- Do NOT describe SQL mechanics (no mention of SELECT, JOIN).
- Focus on medical/business meaning.
- Use provided concepts when possible.
- Output JSON only.
"""
            },
            {
                "role": "user",
                "content": json.dumps(fact, ensure_ascii=False)
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 150   # 🔥 빠르게 제한
        }
    )

    content = response["message"]["content"]
    m = re.search(r"\{.*\}", content, re.DOTALL)

    if not m:
        return {
            "question": question,
            "query": query_sql,
            "bullets": [question[:25], "", ""]
        }

    try:
        summary = json.loads(m.group())
    except Exception:
        return {
            "question": question,
            "query": query_sql,
            "bullets": [question[:25], "", ""]
        }

    # 🔹 강제 고정
    summary["question"] = question
    summary["query"] = query_sql

    bullets = summary.get("bullets", [])
    if not isinstance(bullets, list):
        bullets = []

    summary["bullets"] = (bullets + ["", "", ""])[:3]

    return summary


# 9. MCP
def get_conn():
        return oracledb.connect(
            user=os.getenv('user'),
            password=os.getenv('password'),
            dsn=os.getenv('ORACLE_DSN')
        )

# VALIDATION (MCP POLICY)
def validate_sql(sql: str):
    errors = []

    # SELECT-only 검사
    if not sql.strip().lower().startswith("select"):
        errors.append("SELECT 문만 허용됩니다.")

    # 금지 키워드 검사
    for pattern in FORBIDDEN:
        if re.search(pattern, sql, flags=re.IGNORECASE):
            errors.append(f"금지 키워드 감지: {pattern}")

    return {
        "ok": len(errors) == 0,
        "errors": errors
    }

# DRY RUN (EXPLAIN PLAN)
def dry_run(sql: str):
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("EXPLAIN PLAN FOR " + sql)
        cur.execute("SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY)")
        plan = "\n".join(r[0] for r in cur.fetchall())

        return {
            "ok": True,
            "plan_text": plan
        }

# EXECUTE
def execute_sql(sql: str, limit: int = 1000):
    safe_sql = sql.strip().rstrip(";")

    if "fetch first" not in safe_sql.lower():
        safe_sql = f"{safe_sql}\nFETCH FIRST {limit} ROWS ONLY"

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(safe_sql)

        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(limit)

        return {
            "columns": cols,
            "rows": rows
        }
    

def mcp_pipeline(sql: str):
    report = validate_sql(sql)
    if not report["ok"]:
        print("❌ Validation Failed:", report["errors"])
        return report

    plan = dry_run(sql)


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

def run_pipeline(korean_text: str) -> str:
    # 1. 질문 번역
    result = translate(korean_text)

    # 2. 의료 용어 사전에서 itemid 찾기
    clean_medical_words = kmle_search(result)

    # 3. ChromaDB에서 찾은 단어 찾기
    medical_word_doc, medical_word_meta = concept_retriever(clean_medical_words)

    # 4. ChromaDB에서 관련 테이블 검색
    for_query_table_info = table_retriever(result, medical_word_doc)

    # 5. 사용자 질문과 관련된 column 찾기
    columns = column_selector(result, for_query_table_info)

    # 6. sqlglot을 위한 json 파일 작성
    sql_json = sql_json_builder(result, columns, medical_word_meta)

    #. 7. SQL 쿼리 생성
    sql = sql_renderer(sql_json)

    # 8. SQL 쿼리 조건 & 로직 요약
    summary = summary_json(korean_text, sql, sql_json, medical_word_meta)

    # 9. MCP
    mcp_pipeline(sql)

    return summary


