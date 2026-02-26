import re
import json
from typing import List, Dict, Optional, Tuple, Set
import ollama
import chromadb
import sqlglot
import oracledb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sqlglot import exp

# CHROMA DB 만들 폴더 경로
CHROMA_DB_FOLDER = 'C:\last_project\\for_llm'

class Config:
    """Central configuration for the pipeline"""
    
    # LLM 모델
    EMBEDDING_MODEL = "intfloat/e5-small-v2"
    LLM_MODEL = "qwen2.5:7b"
    
    # ChromaDB 경로
    TABLE_INFO_PATH = f"{CHROMA_DB_FOLDER}\\table_info"
    CONCEPT_PATH = f"{CHROMA_DB_FOLDER}\\concept_store"
    SYNTAX_INFO_PATH = f"{CHROMA_DB_FOLDER}\\syntax_info"
    
    # ChromaDB - Collection names
    TABLE_INFO_COLLECTION = "table_info"
    CONCEPT_COLLECTION = "concept_all"
    SYNTAX_COLLECTION = "syntax_info"
    
    # 의료 약어 JSON
    ABBR_DICT_PATH = "C:\last_project\\for_llm\mimic_iv_abbreviation_160.json"
    
    # Oracle client path
    ORACLE_CLIENT_PATH = r"C:\instant_client\instantclient_21_19"
    
    # Security: Forbidden SQL keywords
    FORBIDDEN_KEYWORDS = [
        r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b",
        r"\bDROP\b", r"\bALTER\b", r"\bTRUNCATE\b",
        r"\bCREATE\b", r"\bGRANT\b", r"\bREVOKE\b"
    ]
    
    # LLM generation settings
    TEMPERATURE = 0.0
    TOP_P = 0.9
    REPEAT_PENALTY = 1.1

    # 해당 단어는 의료 용어가 아니라 테이블 컬럼 이름이기에 예외 처리 수행
    SPECIAL_TERM_MAPPING = {
        "LOS": {
            "table": "icustays",
            "column": "los",
            "description": "Length of stay in ICU in days (float). Direct column, no calculation needed.",
            "table_columns": [
                "subject_id", "hadm_id", "stay_id", "first_careunit",
                "last_careunit", "intime", "outtime", "los"
            ],
            # 번역문에서 이 용어들 중 하나라도 발견되면 매핑 트리거
            "trigger_phrases": ["LOS", "length of stay", "los"]
        },
        # DOD: patients 테이블의 dod 컬럼 (날짜)
        "DOD": {
            "table": "patients",
            "column": "dod",
            "description": "Date of death. NULL if patient is alive.",
            "table_columns": [
                "subject_id", "gender", "anchor_age", "anchor_year",
                "anchor_year_group", "dod"
            ],
            # 번역문에서 이 용어들 중 하나라도 발견되면 매핑 트리거
            "trigger_phrases": ["DOD", "date of death", "dod"]
        },
    }

class PipelineSetup:
    """Initialize all components needed for the pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = None
        self.table_info_collection = None
        self.concept_collection = None
        self.syntax_collection = None
        self.abbr_dict = None
        
    def initialize(self):
        """Initialize all components"""
        print("Initializing pipeline components...")
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        print("Connecting to ChromaDB collections...")
        table_info_client = chromadb.PersistentClient(path=self.config.TABLE_INFO_PATH)
        self.table_info_collection = table_info_client.get_collection(
            self.config.TABLE_INFO_COLLECTION
        )
        
        concept_client = chromadb.PersistentClient(path=self.config.CONCEPT_PATH)
        self.concept_collection = concept_client.get_collection(
            self.config.CONCEPT_COLLECTION
        )
        
        syntax_client = chromadb.PersistentClient(path=self.config.SYNTAX_INFO_PATH)
        self.syntax_collection = syntax_client.get_collection(
            self.config.SYNTAX_COLLECTION
        )
        
        print("Loading medical abbreviation dictionary...")
        with open(self.config.ABBR_DICT_PATH, 'r', encoding='utf-8') as f:
            self.abbr_dict = json.load(f)
        
        print("Initializing Oracle client...")
        load_dotenv('.env')
        oracledb.init_oracle_client(lib_dir=self.config.ORACLE_CLIENT_PATH)
        
        print("Pipeline initialization complete!\n")
    
    def embed_query(self, text: str) -> List[List[float]]:
        """Embed search query"""
        if isinstance(text, str):
            text = [text]
        texts = [f"query: {t}" for t in text]
        return self.embedding_model.encode(
            texts, 
            show_progress_bar=True
        ).tolist()

class TranslationModule:
    """Handles Korean to English translation with medical abbreviation expansion"""
    
    def __init__(
        self,
        abbr_dict: Dict[str, str],
        llm_model: str,
        special_term_mapping: Dict  # [bugfix] 특수 용어 매핑 추가
    ):
        self.abbr_dict = abbr_dict
        self.llm_model = llm_model
        self.special_term_mapping = special_term_mapping  # [bugfix]
        self.reverse_abbr_dict = self._build_reverse_dict()
    
    def _build_reverse_dict(self) -> Dict[str, str]:
        """Build reverse mapping: full term (lowercase) -> abbreviation"""
        return {v.lower(): k for k, v in self.abbr_dict.items()}
    
    def translate(self, korean_text: str) -> str:
        """Translate Korean medical text to English with abbreviation expansion"""
        abbr_text = "\n".join(
            f"- {k} → {v}" for k, v in self.abbr_dict.items()
        )
        
        system_prompt = f"""You are a medical translator specializing in Korean to English translation.

CRITICAL INSTRUCTION: You MUST expand all medical abbreviations using the dictionary provided below.

Medical Abbreviation Dictionary:
{abbr_text}

Translation Rules:
1. Find ALL abbreviations in the Korean text (both Korean and English abbreviations)
2. Replace each abbreviation with its FULL FORM from the dictionary
3. Maintain medical accuracy and clinical context
4. Translate naturally into English
5. Output ONLY the final translated sentence

=== CRITICAL: PRESERVE ALL DATES AND NUMBERS EXACTLY ===
6. NEVER modify dates, times, or numeric values during translation
7. Keep date formats EXACTLY as they appear:
   - "12/1912" → keep as "12/1912" (do NOT change to "December 19, 2012" or any other interpretation)
   - "2012-05" → keep as "2012-05"
   - "10027602" → keep as "10027602" (patient IDs must be exact)
8. Do NOT interpret or reformat dates - preserve the original format
9. Do NOT add assumptions about century or year (1912 is 1912, not 2012)

Example:
Input: "NE 쓰는데 MAP 65도 안 나오는 환자 있어?"
Output: "Is there a patient on norepinephrine whose mean arterial pressure doesn't reach 65?"

Example with dates:
Input: "Since 12/1912, how much did patient 10015931 weigh during the first measurement?"
Output: "Since 12/1912, what was the weight of patient 10015931 during the first measurement?"
(Note: Date "12/1912" is preserved exactly, NOT changed to "December 19, 2012")
"""
        
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": korean_text}
            ],
            options={"temperature": 0, "top_p": 0.9, "repeat_penalty": 1.1}
        )
        
        return response["message"]["content"].strip()
    
    def extract_abbreviations(self, translated_text: str) -> List[str]:
        """
        Extract abbreviations found in the translated text.

        [bugfix] 두 가지 경로로 약어를 탐지:
          1) abbr_dict 역방향 매핑 (full term → abbr)
          2) SPECIAL_TERM_MAPPING의 trigger_phrases 직접 스캔
             → LOS, DOD처럼 abbr_dict에 full term이 없거나 번역문에
               약어 자체(예: 'LOS')가 그대로 남아있을 때도 올바르게 감지.
        """
        text_lower = translated_text.lower()
        found = []

        # ── 경로 1: 기존 abbr_dict 역방향 매핑 ──────────────────────────────
        for full_term, abbr in self.reverse_abbr_dict.items():
            if full_term in text_lower:
                found.append(abbr)

        # ── 경로 2: SPECIAL_TERM_MAPPING trigger_phrases 직접 스캔 ──────────
        # trigger_phrases 중 하나라도 번역문에 포함되면 해당 약어 키를 추가
        for abbr_key, mapping in self.special_term_mapping.items():
            if abbr_key in found:          # 이미 경로 1에서 발견된 경우 스킵
                continue
            triggers = mapping.get("trigger_phrases", [])
            for phrase in triggers:
                if phrase.lower() in text_lower:
                    print(f"    🔑 Special term trigger matched: '{phrase}' → {abbr_key}")
                    found.append(abbr_key)
                    break  # 동일 약어를 중복 추가하지 않음

        return found

class MedicalTermMapper:
    """Maps medical terms to database schema elements"""
    
    def __init__(self, concept_collection, embed_query_fn, special_term_mapping: Dict):
        self.concept_collection = concept_collection
        self.embed_query = embed_query_fn
        # [v4] 특수 용어 예외 매핑 딕셔너리
        self.special_term_mapping = special_term_mapping
    
    def map_term(self, term: str, abbr_dict: Dict[str, str]) -> List[Dict]:
        """Map a single medical term to database metadata"""
        
        # [v4] 특수 용어 예외 처리 - concept store 조회 전에 먼저 확인
        term_upper = term.upper()
        if term_upper in self.special_term_mapping:
            mapping = self.special_term_mapping[term_upper]
            print(f"    ✨ Special term override: {term} → {mapping['table']}.{mapping['column']}")
            # concept store 포맷과 동일한 구조로 반환
            return [{
                "table": mapping["table"],
                "column": mapping["column"],
                "operator": "IS NOT NULL",  # 기본 존재 여부 확인
                "values": "",
                "description": mapping["description"]
            }]
        
        # 일반 용어는 기존대로 concept store에서 조회
        full_term = abbr_dict.get(term, term)
        
        results = self.concept_collection.query(
            query_embeddings=self.embed_query(full_term),
            n_results=1
        )
        
        return results["metadatas"][0]
    
    def map_all_terms(
        self,
        medical_terms: List[str],
        abbr_dict: Dict[str, str],
        translated_text: str   # [bugfix] 한국어 원문 → 번역문 기준으로 변경
    ) -> List[List[Dict]]:
        """
        Map all medical terms that appear in the translated text.

        [bugfix] 기존 코드는 `original_text`(한국어 원문)에 영어 약어가
        포함됐는지 체크했기 때문에, 한국어로만 쓰인 질의에서
        LOS / DOD 등 영어 약어가 스킵되는 문제가 있었음.
        → 이제 `translated_text`(영문 번역 결과)를 기준으로 확인.
        특수 용어(SPECIAL_TERM_MAPPING 키)는 존재 여부 체크 없이
        무조건 매핑.
        """
        mapped = []
        translated_lower = translated_text.lower()

        for term in medical_terms:
            term_upper = term.upper()

            # ── 특수 용어: trigger_phrases로 번역문 포함 여부 확인 ──────────
            if term_upper in self.special_term_mapping:
                triggers = self.special_term_mapping[term_upper].get("trigger_phrases", [])
                triggered = any(p.lower() in translated_lower for p in triggers)
                if not triggered:
                    # trigger phrase가 번역문에 없으면 스킵
                    print(f"  ⏭️ Skipping special term '{term}' (not found in translated text)")
                    continue
            else:
                # ── 일반 용어: 번역문에 full term 또는 약어가 포함됐는지 확인 ──
                full_term = abbr_dict.get(term, term)
                if term.lower() not in translated_lower and full_term.lower() not in translated_lower:
                    print(f"  ⏭️ Skipping term '{term}' (not found in translated text)")
                    continue

            print(f"🔍 Mapping term: {term}")
            metadata = self.map_term(term, abbr_dict)
            mapped.append(metadata)
        
        return mapped

class SchemaSelector:
    """
    Refactored Schema Selector
    - Concept-based
    - Semantic table search
    - Structural heuristic inference
    - Score-based ranking
    """

    def __init__(self, table_info_collection, embed_query_fn, special_term_mapping: Dict):
        self.table_info_collection = table_info_collection
        self.embed_query = embed_query_fn
        self.valid_tables = {}
        self.special_term_mapping = special_term_mapping

    # ============================================================
    # MAIN ENTRY
    # ============================================================
    def select_tables(
        self,
        translated_query: str,
        medical_metadata: List[List[Dict]]
    ) -> List[Dict]:

        print("📊 Step 4: Refactored Schema Selection")

        concept_tables = self._get_tables_from_medical_metadata(medical_metadata)
        semantic_tables = self._semantic_table_search(translated_query)
        structural_tables = self._infer_from_structure(translated_query)

        print(f"  Concept tables: {concept_tables}")
        print(f"  Semantic tables: {semantic_tables}")
        print(f"  Structural tables: {structural_tables}")

        ranked_tables = self._rank_tables(
            translated_query,
            concept_tables,
            semantic_tables,
            structural_tables
        )

        print(f"  Ranked tables: {ranked_tables}")

        selected_tables = self._fetch_table_info(ranked_tables[:5])

        return selected_tables

    # ============================================================
    # 1️⃣ Concept-based tables
    # ============================================================
    def _get_tables_from_medical_metadata(self, medical_metadata):

        tables = list({
            meta[0]["table"]
            for meta in medical_metadata
            if meta and "table" in meta[0]
        })

        return tables

    # ============================================================
    # 2️⃣ Semantic table search (always active)
    # ============================================================
    def _semantic_table_search(self, query_text):

        results = self.table_info_collection.query(
            query_embeddings=self.embed_query(query_text),
            n_results=5
        )

        tables = []
        for meta in results["metadatas"][0]:
            tables.append(meta["table_name"])

        return tables

    # ============================================================
    # 3️⃣ Structural heuristic inference
    # ============================================================
    def _infer_from_structure(self, query_text):

        q = query_text.lower()
        inferred = []

        if "patient" in q or "subject" in q:
            inferred.append("patients")

        if "hospital visit" in q or "admission" in q or "discharg" in q or "marital" in q or "insurance" in q:
            inferred.append("admissions")

        if "icu" in q or "intensive care" in q or "length of stay" in q or "los" in q:
            inferred.append("icustays")

        if any(word in q for word in [
            "weight", "height", "heart rate", "respiratory rate",
            "blood pressure", "temperature", "spo2", "o2 saturation",
            "systolic", "diastolic", "mean arterial"
        ]):
            inferred.append("chartevents")
            inferred.append("d_items")

        if "lab" in q or "blood test" in q or "creatinine" in q or "glucose" in q or "hemoglobin" in q:
            inferred.append("labevents")
            inferred.append("d_labitems")

        if any(word in q for word in ["medication", "drug", "prescri", "dose", "route"]):
            inferred.append("prescriptions")

        if any(word in q for word in ["input", "infus", "fluid", "intake", "iv fluid"]):
            inferred.append("inputevents")
            inferred.append("d_items")

        if any(word in q for word in ["output", "urine", "foley", "drain"]):
            inferred.append("outputevents")
            inferred.append("d_items")

        if any(word in q for word in ["microbiol", "culture", "organism", "specimen", "bacteria"]):
            inferred.append("microbiologyevents")

        if any(word in q for word in ["diagnos", "icd", "condition", "disease"]):
            inferred.append("diagnoses_icd")
            inferred.append("d_icd_diagnoses")

        if any(word in q for word in ["procedure", "surgery", "operation"]):
            inferred.append("procedures_icd")
            inferred.append("d_icd_procedures")

        if any(word in q for word in ["transfer", "care unit", "careunit"]):
            inferred.append("transfers")

        return inferred

    # ============================================================
    # 4️⃣ Ranking logic
    # ============================================================
    def _rank_tables(
        self,
        query_text,
        concept_tables,
        semantic_tables,
        structural_tables
    ):

        score = {}

        # Concept tables → strongest
        for t in concept_tables:
            score[t] = score.get(t, 0) + 3

        # Semantic ranking weight
        for i, t in enumerate(semantic_tables):
            score[t] = score.get(t, 0) + (2 - i * 0.2)

        # Structural inference weight
        for t in structural_tables:
            score[t] = score.get(t, 0) + 2

        ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)

        return [t[0] for t in ranked]

    # ============================================================
    # 5️⃣ Fetch table info + build valid schema cache
    # ============================================================
    def _fetch_table_info(self, tables):

        table_infos = []

        for table in tables:

            # Special mapping 먼저 체크
            special_columns = self._get_special_table_columns(table)
            if special_columns:
                info = {
                    "table_name": table,
                    "column_name": special_columns
                }
                table_infos.append(info)
                self.valid_tables[table] = set(special_columns)
                continue

            results = self.table_info_collection.query(
                query_embeddings=self.embed_query(table),
                n_results=1,
                where={"table_name": table}
            )

            if results["metadatas"][0]:
                info = results["metadatas"][0][0]
                table_infos.append(info)

                columns = info.get("column_name", "[]")
                if isinstance(columns, str):
                    columns = eval(columns)

                self.valid_tables[table] = set(columns)

        return table_infos

    # ============================================================
    # Special term mapping column helper
    # ============================================================
    def _get_special_table_columns(self, table_name: str):

        for term_info in self.special_term_mapping.values():
            if term_info["table"] == table_name and "table_columns" in term_info:
                return term_info["table_columns"]

        return None

    # ============================================================
    # Schema formatting (unchanged)
    # ============================================================
    def format_schema_info(self, table_infos: List[Dict]) -> str:

        schema_text = []

        for info in table_infos:
            table_name = info.get("table_name", "Unknown")
            columns = info.get("column_name", "[]")

            if isinstance(columns, str):
                columns = eval(columns)

            schema_text.append(f"Table: {table_name}")
            schema_text.append(f"Columns: {', '.join(columns)}")
            schema_text.append("")

        return "\n".join(schema_text)
    
class LLMIntentDetector:
    """LLM-based intent detection for accurate query understanding"""
    
    def __init__(self, llm_model: str):
        self.llm_model = llm_model
    
    def detect(self, query_text: str, medical_terms: List[str]) -> Dict:
        """Detect query intent using LLM"""
        
        system_prompt = """You are a SQL query intent analyzer for MIMIC-IV medical databases (Oracle dialect).

Analyze the user's query and determine:
1. Does it need aggregation? (COUNT, AVG, SUM, MAX, MIN, or NONE)
2. What columns need aggregation? (be specific)
3. Is there a comparison condition? (>, <, >=, <=, =, !=)
4. What value is being compared?
5. Should GROUP BY be used? (only for periodic aggregation: daily/monthly/yearly)
6. Is HAVING clause needed? (true if aggregation + comparison on aggregated value)
7. Hospital visit scope: "first", "last", "current", or "all"
8. ICU visit scope: "first_icu", "last_icu", "current_icu", or "all"
9. Time filter: extract date/month/year string if present (e.g. "2100-05", "2100-05-09", "2100")
10. Time filter operator: "=", ">=" or null
11. Is this a top-N ranking query? (needs DENSE_RANK)
12. N value for top-N (number of top results requested)

CRITICAL RULES:
- "mean arterial pressure" is a MEDICAL TERM, not an aggregation request
- "average blood pressure" is a MEDICAL TERM, not an aggregation request
- Only use aggregation if explicitly asking to calculate/compute/find average/count/sum/etc.
- If comparing an aggregated value, MUST use HAVING clause
- WHERE clause: filters rows BEFORE aggregation
- HAVING clause: filters results AFTER aggregation
- "first/last measurement" → NOT aggregation, use subquery with ORDER BY + FETCH FIRST 1 ROWS ONLY
- "daily/monthly/yearly X, take average/max/min" → aggregation WITH GROUP BY by time period
- "since MM/YYYY" → time_filter_op = ">="
- "in MM/YYYY" → time_filter_op = "="
- "current hospital visit" / "this hospital visit" → hospital_visit = "current" (dischtime IS NULL)
- "current ICU" / "this ICU visit" → icu_visit = "current_icu" (outtime IS NULL)

Output ONLY valid JSON:
{
  "aggregation": "COUNT|AVG|SUM|MAX|MIN|NONE",
  "aggregation_column": "column_name or null",
  "comparison_operator": "> | < | >= | <= | = | != or null",
  "comparison_value": "value or null",
  "use_group_by": true|false,
  "group_by_period": "daily|monthly|yearly|null",
  "group_by_columns": ["col1", "col2"] or [],
  "use_having": true|false,
  "use_where": true|false,
  "is_existence_check": true|false,
  "hospital_visit": "first|last|current|all",
  "icu_visit": "first_icu|last_icu|current_icu|all",
  "time_filter": "2100-05|2100-05-09|2100|null",
  "time_filter_op": "=|>=|null",
  "is_top_n": true|false,
  "top_n_value": 3,
  "reasoning": "brief explanation"
}
"""
        
        user_prompt = f"""Query: {query_text}

Medical terms found: {', '.join(medical_terms) if medical_terms else 'None'}

Analyze this query and output JSON:"""
        
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0},
            format="json"
        )
        
        try:
            intent = json.loads(response["message"]["content"])
            return intent
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse intent JSON: {e}")
            return {
                "aggregation": "NONE",
                "aggregation_column": None,
                "comparison_operator": None,
                "comparison_value": None,
                "use_group_by": False,
                "group_by_period": None,
                "group_by_columns": [],
                "use_having": False,
                "use_where": True,
                "is_existence_check": False,
                "hospital_visit": "all",
                "icu_visit": "all",
                "time_filter": None,
                "time_filter_op": None,
                "is_top_n": False,
                "top_n_value": None,
                "reasoning": "Failed to parse, using defaults"
            }

class SQLGenerator:
    """Generates Oracle SQL with strict schema validation and anti-hallucination"""
    
    def __init__(self, llm_model: str, schema_selector: SchemaSelector):
        self.llm_model = llm_model
        self.schema_selector = schema_selector
    
    def format_medical_filters(self, medical_metadata: List[List[Dict]]) -> str:
        """Format medical metadata into SQL WHERE conditions"""
        filters = []
        
        for meta_list in medical_metadata:
            if not meta_list:
                continue
            
            for meta in meta_list:
                table = meta.get("table", "")
                column = meta.get("column", "")
                operator = meta.get("operator", "IN")
                values = meta.get("values", "")
                
                # [v4] values가 비어있는 특수 용어(LOS, DOD)는 필터 조건 생성 생략
                if table and column and values:
                    filters.append(f"{table}.{column} {operator} ({values})")
        
        return "\n".join(filters) if filters else "No specific filters"
    
    def _extract_valid_schema(self, schema_info: str) -> Dict[str, List[str]]:
        """Extract valid tables and columns from schema info"""
        valid_schema = {}
        
        lines = schema_info.strip().split('\n')
        current_table = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Table: '):
                current_table = line.replace('Table: ', '')
                valid_schema[current_table] = []
            elif line.startswith('Columns: ') and current_table:
                cols = line.replace('Columns: ', '').split(', ')
                valid_schema[current_table] = [c.strip() for c in cols]
        
        return valid_schema
    
    def generate(
        self,
        query_text: str,
        schema_info: str,
        medical_metadata: List[List[Dict]],
        intent: Dict
    ) -> str:
        """Generate SQL query with strict validation"""
        
        valid_schema = self._extract_valid_schema(schema_info)
        table_list = list(valid_schema.keys())
        
        schema_constraints = [
            f"Table '{table}' has ONLY these columns: {', '.join(columns)}"
            for table, columns in valid_schema.items()
        ]
        
        # ======================================================================
        # [v5] 포괄적인 Oracle SQL 패턴 가이드 추가
        # ======================================================================
        system_prompt = f"""You are an expert Oracle SQL generator for MIMIC-IV medical databases.

CRITICAL ANTI-HALLUCINATION RULES:
1. Use ONLY these exact table names: {', '.join(table_list)}
2. Do NOT invent table names (NO mimiciii, NO schema prefixes)
3. Do NOT invent column names (NO patient_name, NO diagnosis_name, etc.)
4. EVERY column MUST be prefixed with its table name (e.g., inputevents.subject_id)
5. If multiple tables have the same column, ALWAYS use table prefix to avoid ambiguity

=== CRITICAL: RESPECT USER QUERY EXACTLY (NO MODIFICATIONS) ===
You MUST preserve ALL conditions from the user's question EXACTLY as stated:

1. NEVER modify dates, times, or numeric values
   - "12/1912" means December 1912 → use '1912-12', NOT '2012-12-19'
   - "since 2012" → use >= '2012', NOT >= '2012-01-01'
   - Extract dates EXACTLY from question text

2. NEVER change route names or medical terms
   - "PR route" → use LIKE '%PR%', NOT 'PO'
   - "IM route" → use LIKE '%IM%', NOT corrupted strings
   - Extract exact term from question, do NOT substitute similar terms

3. NEVER add hardcoded conditions not in question
   - Do NOT use if-then-else for specific drug names
   - Generate SQL dynamically based on actual question content
   - No assumptions beyond what user explicitly stated

4. All reasoning must be evidence-based
   - Only use information explicitly stated in the question
   - Do not infer or assume unstated conditions
   - Extract terms verbatim from question text

VALID SCHEMA (USE ONLY THESE):
{chr(10).join(schema_constraints)}

=== ORACLE DATE ARITHMETIC (CRITICAL) ===
Oracle does NOT have DATEDIFF(). Use these patterns instead:
  Days between two dates:   (end_date - start_date)
  ❌ NEVER use: DATEDIFF(), TIMESTAMPDIFF(), DATE_DIFF()
  Length of Stay: use icustays.los directly (float, days) if available
  Hospital LOS: TO_CHAR(CAST(admissions.dischtime AS TIMESTAMP),'%J') - TO_CHAR(CAST(admissions.admittime AS TIMESTAMP),'%J')

=== ORACLE PAGINATION (CRITICAL - NO LIMIT/TOP) ===
Oracle does NOT use LIMIT or TOP. Use:
  FETCH FIRST 1 ROWS ONLY     → get single row
  FETCH FIRST N ROWS ONLY     → get N rows
  Always pair with ORDER BY

=== DATE/TIME FILTERING PATTERNS ===
CRITICAL: Extract dates EXACTLY from user question - do NOT modify or reinterpret:

Examples of CORRECT date parsing:
  "since 10/1912" → '1912-10' (year-month ONLY, use YYYY-MM format)
  "since 12/19/2012" → '2012-12-19' (full date with day specified)
  "in 2015" → '2015' (year only)
  "in May 2015" → '2015-05' (year-month)

CRITICAL RULES FOR DATE FORMATS:
1. If user provides ONLY month/year (e.g., "10/1912", "12/1912"):
   ✅ CORRECT: Use 'YYYY-MM' format → '1912-10', '1912-12'
   ❌ WRONG: Add arbitrary day like '1912-10-19' or '1912-10-01'
   
2. If user provides full date with day (e.g., "12/19/2012", "10/1/2015"):
   ✅ CORRECT: Use 'YYYY-MM-DD' format → '2012-12-19', '2015-10-01'
   
3. NEVER add day numbers (01, 19, etc.) unless explicitly in the question
4. Match TO_CHAR format to the date precision provided:
   - Month/Year only → TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM') >= 'YYYY-MM'
   - Full date → TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM-DD') >= 'YYYY-MM-DD'

Date format patterns (use appropriate format based on extracted date):
  Month filter:   TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM') = 'YYYY-MM'
  Month since:    TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM') >= 'YYYY-MM'
  Date filter:    TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM-DD') = 'YYYY-MM-DD'
  Date since:     TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM-DD') >= 'YYYY-MM-DD'
  Year filter:    TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY') = 'YYYY'
  Year since:     TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY') >= 'YYYY'


=== HOSPITAL VISIT PATTERNS ===
Current (ongoing) hospital visit:    admissions.dischtime IS NULL
Completed hospital visits:           NOT admissions.dischtime IS NULL
First hospital visit:
  hadm_id IN (SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = X
              AND NOT admissions.dischtime IS NULL ORDER BY admissions.admittime ASC NULLS FIRST FETCH FIRST 1 ROWS ONLY)
Last hospital visit:
  hadm_id IN (SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = X
              AND NOT admissions.dischtime IS NULL ORDER BY admissions.admittime DESC NULLS LAST FETCH FIRST 1 ROWS ONLY)

=== ICU VISIT PATTERNS ===
Current ICU stay:    icustays.outtime IS NULL
First ICU stay:
  stay_id IN (SELECT icustays.stay_id FROM icustays WHERE icustays.hadm_id IN (...)
              AND NOT icustays.outtime IS NULL ORDER BY icustays.intime ASC NULLS FIRST FETCH FIRST 1 ROWS ONLY)
Last ICU stay:
  stay_id IN (SELECT icustays.stay_id FROM icustays WHERE icustays.hadm_id IN (...)
              AND NOT icustays.outtime IS NULL ORDER BY icustays.intime DESC NULLS LAST FETCH FIRST 1 ROWS ONLY)

=== DRUG / MEDICATION / ROUTE PATTERNS (CRITICAL) ===
ALWAYS extract exact terms from the user's question and use LIKE pattern matching:

For drug names from question:
  Question contains "insulin" → UPPER(prescriptions.drug) LIKE UPPER('%insulin%')
  Question contains "midazolam" → UPPER(prescriptions.drug) LIKE UPPER('%midazolam%')
  ✅ CORRECT: Extract drug name from question, use LIKE '%drugname%'
  ❌ WRONG: Use = 'exact_match' (always use LIKE for fuzzy matching)
  ❌ WRONG: Hardcode drug names not mentioned in question

For administration routes from question:
  Question: "via PR route" → prescriptions.route LIKE '%PR%'
  Question: "via IM route" → prescriptions.route LIKE '%IM%'
  Question: "via PO route" → prescriptions.route LIKE '%PO%'
  ✅ CORRECT: Extract exact route abbreviation from question
  ❌ WRONG: Change 'PR' to 'PO' or corrupt 'IM' to other strings
  ❌ WRONG: Substitute different route not in question

For measurement labels in d_items/d_labitems:
  Similarly use LIKE '%label%' for fuzzy matching based on question text.


=== FIRST/LAST VALUE PATTERNS ===
To get first or last measured value for a patient:
  chartevents.charttime = (
    SELECT DISTINCT chartevents.charttime FROM chartevents
    WHERE <same filters>
    ORDER BY chartevents.charttime ASC NULLS FIRST   -- for first
    FETCH FIRST 1 ROWS ONLY
  )
Same pattern applies to labevents.charttime, prescriptions.starttime, outputevents.charttime, inputevents.starttime, microbiologyevents.charttime

=== TOP-N RANKING PATTERN ===
Use DENSE_RANK() window function for top-N queries:
  SELECT T1.drug FROM (
    SELECT prescriptions.drug,
           DENSE_RANK() OVER (ORDER BY COUNT(*) DESC NULLS LAST) AS C1
    FROM prescriptions
    GROUP BY prescriptions.drug
  ) T1 WHERE T1.C1 <= N

=== ITEM LOOKUP PATTERNS ===
For chartevents/inputevents/outputevents items use LIKE in d_items lookup:
  chartevents.itemid IN (SELECT d_items.itemid FROM d_items
    WHERE UPPER(d_items.label) LIKE UPPER('%heart rate%') AND d_items.linksto = 'chartevents')
For lab items:
  labevents.itemid IN (SELECT d_labitems.itemid FROM d_labitems
    WHERE UPPER(d_labitems.label) LIKE UPPER('%glucose%'))

=== AGGREGATION RULES ===
1. Ambiguous columns: ALWAYS prefix with table name
2. GROUP BY: Use exact same table-prefixed columns as in SELECT
3. ❌ WRONG: WHERE AVG(value) < 65
   ✅ CORRECT: GROUP BY ... HAVING AVG(table.value) < 65
4. For min/max value WITH timestamp (when did patient have max X?):
   ORDER BY labevents.valuenum DESC NULLS LAST, labevents.charttime DESC NULLS LAST FETCH FIRST 1 ROWS ONLY

=== PERIODIC AGGREGATION PATTERN (daily/monthly/yearly) ===
"daily X, take average/min/max" → GROUP BY TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM-DD')
"monthly X, take average"       → GROUP BY TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY-MM')
"yearly X, take max"            → GROUP BY TO_CHAR(CAST(col AS TIMESTAMP), 'YYYY')

Return ONLY the SQL query, no explanations.
DO NOT include any comments in the SQL (no --, no /* */, no explanations)."""

        medical_filters = self.format_medical_filters(medical_metadata)
        
        intent_guidance = f"""QUERY INTENT ANALYSIS:
- Aggregation: {intent.get('aggregation', 'NONE')}
- Aggregation Column: {intent.get('aggregation_column', 'N/A')}
- Comparison: {intent.get('comparison_operator', 'N/A')} {intent.get('comparison_value', '')}
- Use GROUP BY: {intent.get('use_group_by', False)}
- Group By Period: {intent.get('group_by_period', 'null')}
- Use HAVING: {intent.get('use_having', False)}
- Use WHERE: {intent.get('use_where', True)}
- Is Existence Check: {intent.get('is_existence_check', False)}
- Hospital Visit Scope: {intent.get('hospital_visit', 'all')}
- ICU Visit Scope: {intent.get('icu_visit', 'all')}
- Time Filter: {intent.get('time_filter', 'null')} (op: {intent.get('time_filter_op', 'null')})
- Top-N Query: {intent.get('is_top_n', False)} (N={intent.get('top_n_value', 'null')})
- Reasoning: {intent.get('reasoning', '')}
"""
        
        user_prompt = f"""Generate an Oracle SQL query for:

QUESTION:
{query_text}

AVAILABLE SCHEMA (EXACT TABLE AND COLUMN NAMES):
{schema_info}

MANDATORY FILTERS (include in WHERE clause if applicable):
{medical_filters}

{intent_guidance}

CRITICAL REMINDERS:
1. Every column MUST have table prefix (table.column)
2. Use ONLY tables and columns listed in schema above
3. NO schema prefixes like mimiciii
4. NO invented columns
5. NO DATEDIFF() / LIMIT / TOP — Oracle uses (date2 - date1) and FETCH FIRST N ROWS ONLY
6. If aggregation + comparison, use HAVING not WHERE
7. GROUP BY columns must match SELECT columns (with table prefix)
8. EXTRACT EXACT TERMS FROM QUESTION:
   - Drug names: Extract from question → use UPPER(prescriptions.drug) LIKE UPPER('%drugname%')
   - Routes: Extract from question → use prescriptions.route LIKE '%ROUTE%'
   - Dates: Parse EXACTLY as in question with CORRECT format:
     * "since 10/1912" → TO_CHAR(..., 'YYYY-MM') >= '1912-10' (NO DAY NUMBER)
     * "since 12/19/2012" → TO_CHAR(..., 'YYYY-MM-DD') >= '2012-12-19' (day specified)
     * NEVER add day numbers (01, 19, etc.) unless explicitly in question
   - Do NOT modify, substitute, or corrupt any terms from the question
9. For d_items/d_labitems label lookups: use LIKE '%label%' for fuzzy matching
10. For first/last value: use ORDER BY + FETCH FIRST 1 ROWS ONLY on timestamp column
11. Current hospital visit: admissions.dischtime IS NULL
    First hospital visit: ORDER BY admissions.admittime ASC NULLS FIRST FETCH FIRST 1 ROWS ONLY
    Last hospital visit: ORDER BY admissions.admittime DESC NULLS LAST FETCH FIRST 1 ROWS ONLY
12. For top-N ranking: use DENSE_RANK() OVER (ORDER BY COUNT(*) DESC NULLS LAST)
13. NO HARDCODED CONDITIONS - generate SQL dynamically based on question content only


Generate the SQL query now:"""
        
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0}
        )
        
        sql = response["message"]["content"].strip()
        sql = self._clean_sql(sql)
        
        # [v4] DATEDIFF 패턴 자동 수정
        sql = self._fix_oracle_date_syntax(sql)
        
        # [v5] prescriptions.drug = '...' → LIKE '%...%' 변환
        sql = self._fix_prescription_drug_filter(sql)
        
        # Schema 검증
        validation_errors = self._validate_against_schema(sql, valid_schema)
        if validation_errors:
            print("\n⚠️ Schema validation errors found:")
            for error in validation_errors:
                print(f"  - {error}")
            print("\n🔧 Attempting to fix...")
            sql = self._fix_sql(sql, valid_schema, validation_errors)
        
        return sql
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and format SQL"""
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        sql = ' '.join(sql.split())
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    def _fix_prescription_drug_filter(self, sql: str) -> str:
        """
        [v5] prescriptions.drug = '약물명' 패턴을
        UPPER(prescriptions.drug) LIKE UPPER('%약물명%') 로 변환.
        
        단, 이미 LIKE 가 포함된 조건은 그대로 유지.
        """
        import re
        original = sql

        # prescriptions.drug = '...' 또는 prescriptions.drug= '...' 패턴 탐지
        # 단, 이미 LIKE가 사용된 경우 제외
        def replace_drug_eq(m):
            drug_name = m.group(1)
            return f"UPPER(prescriptions.drug) LIKE UPPER('%{drug_name}%')"

        # = '값' 패턴에서 LIKE가 없는 경우만 치환
        sql = re.sub(
            r"\bprescriptions\.drug\s*=\s*'([^']+)'",
            replace_drug_eq,
            sql,
            flags=re.IGNORECASE
        )

        if sql != original:
            print(f"    🔧 [v5] Drug filter converted to LIKE:")
            print(f"       Before: {original[:150]}..." if len(original) > 150 else f"       Before: {original}")
            print(f"       After:  {sql[:150]}..." if len(sql) > 150 else f"       After:  {sql}")

        return sql

    def _fix_oracle_date_syntax(self, sql: str) -> str:
        """
        [v4] DATEDIFF / TIMESTAMPDIFF 등 비Oracle 날짜 함수를 Oracle 문법으로 자동 변환.
        
        패턴 예시:
          DATEDIFF(storetime, charttime)         → (storetime - charttime)
          DATEDIFF(storetime - charttime)        → (storetime - charttime)  ← 기존 잘못된 단일 인자
          TIMESTAMPDIFF(DAY, date1, date2)       → (date2 - date1)
        """
        original = sql
        
        # 패턴 1: DATEDIFF(expr1, expr2) - 두 인자 형태
        sql = re.sub(
            r'\bDATEDIFF\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)',
            r'(\1 - \2)',
            sql,
            flags=re.IGNORECASE
        )
        
        # 패턴 2: DATEDIFF(expr1 - expr2) - 단일 인자에 빼기 연산 포함 (잘못된 형태)
        sql = re.sub(
            r'\bDATEDIFF\s*\(\s*([^)]+?\s*-\s*[^)]+?)\s*\)',
            r'(\1)',
            sql,
            flags=re.IGNORECASE
        )
        
        # 패턴 3: TIMESTAMPDIFF(unit, date1, date2)
        sql = re.sub(
            r'\bTIMESTAMPDIFF\s*\(\s*\w+\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)',
            r'(\2 - \1)',
            sql,
            flags=re.IGNORECASE
        )
        
        if sql != original:
            print(f"    🔧 [v4] Date function auto-fixed:")
            print(f"       Before: {original[:100]}..." if len(original) > 100 else f"       Before: {original}")
            print(f"       After:  {sql[:100]}..." if len(sql) > 100 else f"       After:  {sql}")
        
        return sql
    
    def _validate_against_schema(self, sql: str, valid_schema: Dict[str, List[str]]) -> List[str]:
        """Validate SQL against actual schema"""
        errors = []
        sql_upper = sql.upper()
        
        # Check for schema prefixes
        if re.search(r'\b[A-Z_]+\.(?:' + '|'.join([t.upper() for t in valid_schema.keys()]) + r')\b', sql_upper):
            errors.append("Found schema prefix (like mimiciii.) - use bare table names only")
        
        # Check for hallucinated columns
        hallucinated = [
            'PATIENT_NAME', 'PATIENT_ID', 'FULL_NAME',
            'DIAGNOSIS_NAME', 'MEDICATION_NAME', 'DOCTOR_NAME',
            'ICU_NAME', 'LENGTH_OF_STAY'
        ]
        
        for col in hallucinated:
            col_exists = any(
                col.lower() in [c.lower() for c in columns]
                for columns in valid_schema.values()
            )
            if not col_exists and col in sql_upper:
                errors.append(f"Column '{col}' not found in schema (hallucination)")
        
        # [v4] DATEDIFF가 여전히 남아있으면 에러 표시
        if re.search(r'\bDATEDIFF\b', sql_upper):
            errors.append("DATEDIFF is not supported in Oracle. Use (date2 - date1) instead.")
        
        # Check for ambiguous columns in multi-table queries
        if 'JOIN' in sql_upper:
            unqualified = re.findall(
                r'(?:SELECT|WHERE|GROUP BY|HAVING|ON)\s+(?:[^,\s]+\s+)?(?!AVG|COUNT|SUM|MAX|MIN)([a-z_]+)(?:\s|,|;|\)|$)',
                sql, re.IGNORECASE
            )
            
            for col in unqualified:
                tables_with_col = [
                    table for table, columns in valid_schema.items()
                    if col.lower() in [c.lower() for c in columns]
                ]
                if len(tables_with_col) > 1:
                    errors.append(f"Column '{col}' is ambiguous (exists in {len(tables_with_col)} tables)")
        
        return errors
    
    def _fix_sql(self, sql: str, valid_schema: Dict[str, List[str]], errors: List[str]) -> str:
        """Attempt to fix common SQL errors"""
        fixed_sql = sql
        
        # Fix schema prefixes
        for table in valid_schema.keys():
            fixed_sql = re.sub(
                r'\b[a-z_]+\.' + table + r'\b',
                table,
                fixed_sql,
                flags=re.IGNORECASE
            )
        
        return fixed_sql

class SQLValidator:
    """Validates SQL for syntax and security"""
    
    def __init__(self, forbidden_keywords: List[str]):
        self.forbidden_keywords = forbidden_keywords
    
    def validate_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax using sqlglot"""
        try:
            sqlglot.parse_one(sql, dialect="oracle")
            return True, None
        except Exception as e:
            return False, str(e)
    
    def validate_security(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Check for forbidden SQL operations"""
        sql_upper = sql.upper()
        
        for pattern in self.forbidden_keywords:
            if re.search(pattern, sql_upper):
                matched = re.search(pattern, sql_upper).group()
                return False, f"Forbidden keyword detected: {matched}"
        
        return True, None
    
    def validate_aggregation_usage(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Check for proper aggregation function usage"""
        sql_upper = sql.upper()
        agg_functions = ['AVG', 'COUNT', 'SUM', 'MAX', 'MIN']
        
        if 'WHERE' in sql_upper:
            where_clause = sql_upper.split('WHERE')[1].split('GROUP BY')[0] \
                if 'GROUP BY' in sql_upper else sql_upper.split('WHERE')[1]
            where_clause = where_clause.split('HAVING')[0] if 'HAVING' in where_clause else where_clause
            
            for func in agg_functions:
                if func + '(' in where_clause:
                    return False, f"{func} function cannot be used in WHERE clause. Use HAVING instead."
        
        return True, None
    
    def validate(self, sql: str) -> Dict:
        """Perform complete validation"""
        result = {
            "valid": False,
            "syntax_valid": False,
            "security_valid": False,
            "aggregation_valid": False,
            "errors": []
        }
        
        syntax_valid, syntax_error = self.validate_syntax(sql)
        result["syntax_valid"] = syntax_valid
        if not syntax_valid:
            result["errors"].append(f"Syntax error: {syntax_error}")
        
        security_valid, security_error = self.validate_security(sql)
        result["security_valid"] = security_valid
        if not security_valid:
            result["errors"].append(f"Security error: {security_error}")
        
        agg_valid, agg_error = self.validate_aggregation_usage(sql)
        result["aggregation_valid"] = agg_valid
        if not agg_valid:
            result["errors"].append(f"Aggregation error: {agg_error}")
        
        result["valid"] = syntax_valid and security_valid and agg_valid
        
        return result

class MedicalTextToSQLPipeline:
    """Complete end-to-end pipeline with LLM intent detection"""
    
    def __init__(
        self,
        translator: TranslationModule,
        mapper: MedicalTermMapper,
        schema_selector: SchemaSelector,
        intent_detector: LLMIntentDetector,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator
    ):
        self.translator = translator
        self.mapper = mapper
        self.schema_selector = schema_selector
        self.intent_detector = intent_detector
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
    
    def process(self, korean_query: str) -> Dict:
        """Process a Korean medical query through the complete pipeline"""
        
        print("\n" + "="*80)
        print("🚀 STARTING PIPELINE V5 (LIKE Drug Filter + Enhanced Intent + Oracle Patterns)")
        print("="*80)
        
        result = {
            "original_query": korean_query,
            "translated_query": None,
            "medical_terms": [],
            "medical_metadata": [],
            "selected_tables": [],
            "intent": {},
            "generated_sql": None,
            "validation": {},
            "success": False
        }
        
        try:
            # Step 1: Translation
            print("\n Step 1: Translation")
            print(f"  Input: {korean_query}")
            translated = self.translator.translate(korean_query)
            result["translated_query"] = translated
            print(f"  Output: {translated}")
            
            # Step 2: Extract medical terms
            # [bugfix] 번역문을 기준으로 약어 추출 (기존과 동일하지만 이제
            #          SPECIAL_TERM_MAPPING trigger_phrases도 함께 스캔)
            print("\n Step 2: Medical Term Extraction")
            medical_terms = self.translator.extract_abbreviations(translated)
            result["medical_terms"] = medical_terms
            print(f"  Found: {medical_terms}")
            
            # Step 3: Map to database
            # [bugfix] original_text → translated (영문 번역 결과 기준으로 매핑)
            print("\nStep 3: Medical Term Mapping")
            medical_metadata = self.mapper.map_all_terms(
                medical_terms,
                self.translator.abbr_dict,
                translated          # ← [bugfix] 한국어 원문 대신 번역문 전달
            )
            result["medical_metadata"] = medical_metadata
            
            # Step 4: Select schema
            print("\n Step 4: Schema Selection")
            selected_tables = self.schema_selector.select_tables(
                translated,
                medical_metadata
            )
            result["selected_tables"] = selected_tables
            schema_info = self.schema_selector.format_schema_info(selected_tables)
            
            # Step 5: LLM Intent Detection
            print("\n Step 5: LLM Intent Detection")
            intent = self.intent_detector.detect(translated, medical_terms)
            result["intent"] = intent
            print(f"  Aggregation: {intent.get('aggregation')}")
            print(f"  Use HAVING: {intent.get('use_having')}")
            print(f"  Reasoning: {intent.get('reasoning')}")
            
            # Step 6: Generate SQL
            print("\n Step 6: SQL Generation")
            sql = self.sql_generator.generate(
                translated,
                schema_info,
                medical_metadata,
                intent
            )
            result["generated_sql"] = sql
            print(f"  Generated SQL:")
            print(f"  {sql}")
            
            # Step 7: Validate
            print("\n Step 7: Validation")
            validation = self.sql_validator.validate(sql)
            result["validation"] = validation
            result["success"] = validation["valid"]
            
            if validation["valid"]:
                print("SQL is valid and safe!")
            else:
                print("SQL validation failed:")
                for error in validation["errors"]:
                    print(f"    - {error}")
        
        except Exception as e:
            print(f"\n Pipeline error: {str(e)}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80 + "\n")
        
        return result

def generate_medical_sql_explanation_json(
    question_text,
    sql_structure_json,
    execution_summary=None
):

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": f"""
You are a medical explanation generator for clinicians.

CRITICAL RULES:
- Output MUST be written entirely in Korean.
- Return ONLY valid JSON.
- Maximum 60 Korean words.
- Use simple, clinical-friendly language.
- Assume the reader does NOT know SQL or database concepts.
- DO NOT mention technical terms such as:
  WHERE, JOIN, DISTINCT, GROUP BY, COUNT, SELECT, table, column.
- Do NOT describe database logic.
- Explain only:
  1) 어떤 환자 또는 입실 데이터를 대상으로 했는지
  2) 무엇을 계산했는지
- Keep it short, clear, and clinically intuitive.
- No clinical interpretation.
- No added assumptions.

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

SQL:
{json.dumps(sql_structure_json, indent=2)}

Execution Summary:
{execution_summary if execution_summary else "None provided."}

Generate concise explanation SQL in Korean.
"""
            }
        ],
        options={"temperature": 0}
    )

    json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
    explanation_json = json.loads(json_match.group())

    return explanation_json['explanation']

def partial_mask_value(value, show_front=4, mask_char="*"):
    if value is None:
        return None

    value_str = str(value)

    if len(value_str) <= show_front:
        return mask_char * len(value_str)

    return value_str[:show_front] + mask_char * (len(value_str) - show_front)

def auto_mask_mimic_partial(result: dict,
                            show_front=4,
                            mask_char="*",
                            custom_sensitive_cols=None) -> dict:
    MIMIC_SENSITIVE_COLUMNS = {
        "SUBJECT_ID",
        "HADM_ID",
        "STAY_ID",
        "ICUSTAY_ID",
        "ROW_ID"
    }
    
    columns = result.get("columns", [])
    rows = result.get("rows", [])

    # 기본 + 사용자 확장 컬럼
    sensitive_cols = set(MIMIC_SENSITIVE_COLUMNS)
    if custom_sensitive_cols:
        sensitive_cols.update(custom_sensitive_cols)

    # 실제 존재하는 컬럼만 필터링
    col_index_map = {col: idx for idx, col in enumerate(columns)}
    mask_indexes = [
        col_index_map[col]
        for col in sensitive_cols
        if col in col_index_map
    ]

    if not mask_indexes:
        return result  # 마스킹할 컬럼 없으면 그대로 반환

    masked_rows = []
    for row in rows:
        row_list = list(row)

        for idx in mask_indexes:
            row_list[idx] = partial_mask_value(
                row_list[idx],
                show_front=show_front,
                mask_char=mask_char
            )

        masked_rows.append(tuple(row_list))

    return {
        "columns": columns,
        "rows": masked_rows
    }


config = Config()
setup = PipelineSetup(config)
setup.initialize()

translator = TranslationModule(
    setup.abbr_dict,
    config.LLM_MODEL,
    config.SPECIAL_TERM_MAPPING
)

mapper = MedicalTermMapper(
    setup.concept_collection, 
    setup.embed_query,
    config.SPECIAL_TERM_MAPPING  # [v4] 특수 용어 매핑 전달
)

schema_selector = SchemaSelector(
    setup.table_info_collection, 
    setup.embed_query,
    config.SPECIAL_TERM_MAPPING  # [v4]
)

intent_detector = LLMIntentDetector(config.LLM_MODEL)

sql_generator = SQLGenerator(config.LLM_MODEL, schema_selector)

sql_validator = SQLValidator(config.FORBIDDEN_KEYWORDS)

pipeline = MedicalTextToSQLPipeline(
    translator=translator,
    mapper=mapper,
    schema_selector=schema_selector,
    intent_detector=intent_detector,
    sql_generator=sql_generator,
    sql_validator=sql_validator
)

def change_bind_query(query):
    """
    SQL 쿼리의 리터럴 값을 바인드 변수로 변환합니다.
    모든 날짜 정보는 원본 그대로 유지됩니다.
    """
    try:
        parsed = sqlglot.parse_one(query, dialect="oracle")

        bind_values = {}
        bind_idx = 1

        for literal in parsed.find_all(exp.Literal):
            parent = literal.parent
            
            # TO_CHAR 등의 format 인자는 스킵
            if parent and "format" in parent.args and parent.args["format"] is literal:
                continue

            bind_name = f"v{bind_idx}"
            bind_idx += 1

            if literal.is_string:
                # 문자열 값을 그대로 사용 (날짜 변환 없음)
                bind_values[bind_name] = literal.this

            elif literal.is_number:
                bind_values[bind_name] = float(literal.this)

            # Literal → Parameter 로 교체
            literal.replace(exp.Parameter(this=bind_name))

        res = parsed.sql(dialect="oracle").replace("@", ":")
        return res, bind_values
    except Exception as e:
        print(f"Bind query conversion failed: {e}")
        return query, {}

def llm_answer(korean_text):
    result = pipeline.process(korean_text)
    sql = result['generated_sql']
    bind_query, bind_dict = change_bind_query(sql)
    llm_answer = generate_medical_sql_explanation_json(result['original_query'], result['generated_sql'])
    return sql, bind_query, bind_dict, llm_answer
