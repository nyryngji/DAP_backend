import pandas as pd
from passlib.context import CryptContext
import sqlparse
import plotly.graph_objects as go
import pandas as pd
from sqlglot import parse_one
from sqlglot.expressions import Table
import io

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)

# 로그인 기능
def login_with_db(con, user_id, user_password):
    cursor = con.cursor()

    sql = """
        SELECT USER_SEQ, USER_NAME, ADMIN_ROLE, USER_PASSWORD, LAST_LOGIN_TIME
        FROM LOGIN_SESSION
        WHERE USER_ID = :USER_ID
    """

    cursor.execute(sql, {"USER_ID": user_id})

    row = cursor.fetchone()

    if not row:
        cursor.close()
        return None

    columns = [col[0] for col in cursor.description]
    result = dict(zip(columns, row))

    cursor.close()

    if pwd_context.verify(user_password, result["USER_PASSWORD"]):
        return {
            "USER_SEQ": result["USER_SEQ"],
            "USER_NAME": result["USER_NAME"],
            "ADMIN_ROLE": result["ADMIN_ROLE"],
            "LAST_LOGIN_TIME": result["LAST_LOGIN_TIME"]
        }
        
# 채팅 세션 정보 저장
def save_chat_session(cursor, conn, CHAT_SESSION_ID, USER_SEQ):
    sql = """
        INSERT INTO CHAT_SESSION (CHAT_SESSION_ID, USER_SEQ, STARTED_AT)
        VALUES (:CHAT_SESSION_ID, :USER_SEQ, SYSDATE)
    """

    try:
        cursor.execute(
            sql,
            {
                "CHAT_SESSION_ID": CHAT_SESSION_ID,
                "USER_SEQ": USER_SEQ
            }
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("❌ DB Error 발생:", e)
        raise

# 사용자 질문 저장
def save_user_question(cursor, conn, CHAT_SESSION_ID, QUESTION_TEXT):
    sql = """
    INSERT INTO CHAT_HISTORY (CHAT_SESSION_ID, QUESTION_TIME, QUESTION_TEXT)
    VALUES (:CHAT_SESSION_ID, SYSDATE, :QUESTION_TEXT)
    """
    try:
        cursor.execute(
            sql,
            {
                "CHAT_SESSION_ID": CHAT_SESSION_ID,
                "QUESTION_TEXT": QUESTION_TEXT
            }
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.commit()

# LLM 답변 저장
def save_llm_answer(cursor, conn, CHAT_SESSION_ID, ANSWER_TEXT):
    QUESTION_ID = cursor.var(int)

    sql = """
    INSERT INTO CHAT_HISTORY (CHAT_SESSION_ID, ANSWER_TIME, ANSWER_TEXT)
    VALUES (:CHAT_SESSION_ID, SYSDATE, :ANSWER_TEXT)
    RETURNING QUESTION_ID INTO :QUESTION_ID
    """

    try:
        cursor.execute(
            sql,
            {
                "CHAT_SESSION_ID": CHAT_SESSION_ID,
                "ANSWER_TEXT": ANSWER_TEXT,
                "QUESTION_ID": QUESTION_ID,
            }
        )
        conn.commit()
        return QUESTION_ID.getvalue()[0]
    except Exception as e:
        conn.rollback()
        raise e

def update_llm_answer(cursor, conn, ANSWER_TEXT, QUESTION_ID):
    sql = '''UPDATE CHAT_HISTORY SET ANSWER_TEXT = :ANSWER_TEXT WHERE QUESTION_ID = :QUESTION_ID'''
    try:
        cursor.execute(
            sql,
            {
                "ANSWER_TEXT": ANSWER_TEXT,
                "QUESTION_ID": QUESTION_ID
            }
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e

# 채팅 내역 불러오기
def load_chat_history(conn, CHAT_SESSION_ID):
    cursor = conn.cursor()

    sql = """
        SELECT QUESTION_ID, QUESTION_TEXT, ANSWER_TEXT
        FROM CHAT_HISTORY
        WHERE CHAT_SESSION_ID = :CHAT_SESSION_ID
        ORDER BY QUESTION_ID
    """

    cursor.execute(sql, {"CHAT_SESSION_ID": CHAT_SESSION_ID})

    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()

    cursor.close()

    return [dict(zip(columns, row)) for row in rows]

def make_table_from_sql(conn, sql_text, bind_values, limit=5):
    """SQL 실행하고 DataFrame 반환"""
    cur = conn.cursor()
    
    try:
        # ✅ Oracle에서 limit 처리
        limited_sql = f"""
        SELECT * FROM (
            {sql_text}
        ) WHERE ROWNUM <= {limit}
        """
        
        cur.execute(limited_sql, bind_values)
        
        # ✅ limit만큼만 fetch
        rows = cur.fetchmany(limit)
        columns = [desc[0] for desc in cur.description]
        
        return {
            "columns": columns,
            "rows": rows
        }
    
    finally:
        cur.close()

def save_sql(cursor, conn, USER_SEQ, CHAT_SESSION_ID, QUESTION_ID, SQL_TEXT, BIND_VALUES, CREATING_TIME):
    
    sql = """
    INSERT INTO SAVED_QUERY (USER_SEQ, CHAT_SESSION_ID, QUESTION_ID, SQL_TEXT, CREATED_AT, BIND_VALUES, CREATING_TIME)
    VALUES (:USER_SEQ, :CHAT_SESSION_ID, :QUESTION_ID, :SQL_TEXT, SYSTIMESTAMP, :BIND_VALUES, :CREATING_TIME)
    """
    try:
        cursor.execute(
            sql,
            {
                "USER_SEQ" : USER_SEQ, 
                "CHAT_SESSION_ID" : CHAT_SESSION_ID, 
                "QUESTION_ID" : QUESTION_ID, 
                "SQL_TEXT" : SQL_TEXT,
                "BIND_VALUES" : BIND_VALUES,
                "CREATING_TIME" : CREATING_TIME
            }
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    
def load_sql_for_table_info(conn, QUESTION_ID):
    cursor = conn.cursor()

    sql = """
        SELECT CHAT_SESSION_ID, SQL_TEXT, BIND_VALUES
        FROM SAVED_QUERY
        WHERE QUESTION_ID = :QUESTION_ID
    """

    cursor.execute(sql, {"QUESTION_ID": QUESTION_ID})
    row = cursor.fetchone()

    cursor.close()

    if not row:
        return None

    columns = ["CHAT_SESSION_ID", "SQL_TEXT", "BIND_VALUES"]

    return dict(zip(columns, row))
    
def user_chat_list(conn, USER_SEQ):
    cursor = conn.cursor()

    sql = '''
        SELECT
            ch.CHAT_SESSION_ID,
            ch.QUESTION_TEXT,
            ch.QUESTION_TIME
        FROM (
            SELECT
                CHAT_SESSION_ID,
                QUESTION_TEXT,
                QUESTION_TIME,
                ROW_NUMBER() OVER (
                    PARTITION BY CHAT_SESSION_ID
                    ORDER BY QUESTION_TIME
                ) RN
            FROM CHAT_HISTORY
            WHERE QUESTION_TEXT IS NOT NULL
        ) ch
        JOIN CHAT_SESSION cs
            ON ch.CHAT_SESSION_ID = cs.CHAT_SESSION_ID
        WHERE cs.USER_SEQ = :USER_SEQ
          AND ch.RN = 1
        ORDER BY ch.QUESTION_TIME DESC
    '''

    cursor.execute(sql, {"USER_SEQ": USER_SEQ})

    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()

    cursor.close()

    return [dict(zip(columns, row)) for row in rows]

def sql_parsing(raw_sql):
    formatted_sql = sqlparse.format(
        raw_sql,
        reindent=True,          # 들여쓰기 자동 정리
        keyword_case="upper"    # SELECT, FROM 등 대문자
    )

    return formatted_sql.strip()

def update_favorite(cursor, conn, TITLE, QUESTION_ID):
    sql = '''UPDATE SAVED_QUERY 
            SET
                IS_FAVORITE = 'Y',
                TITLE = :TITLE
            WHERE QUESTION_ID = :QUESTION_ID'''
    try:
        cursor.execute(
            sql,
            {
                "TITLE": TITLE,
                "QUESTION_ID": QUESTION_ID
            }
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    
def isit_favorite(conn, QUESTION_ID):
    cursor = conn.cursor()

    sql = """
        SELECT TITLE, IS_FAVORITE
        FROM SAVED_QUERY
        WHERE QUESTION_ID = :QUESTION_ID
    """

    cursor.execute(sql, {"QUESTION_ID": QUESTION_ID})
    row = cursor.fetchone()

    cursor.close()

    if not row:
        return None

    columns = ["TITLE", "IS_FAVORITE"]
    return dict(zip(columns, row))

def total_favorite_list_len(conn, USER_SEQ):
    cursor = conn.cursor()

    sql = """
        SELECT COUNT(1)
        FROM SAVED_QUERY
        WHERE USER_SEQ = :USER_SEQ
          AND IS_FAVORITE = 'Y'
    """

    cursor.execute(sql, {"USER_SEQ": USER_SEQ})
    row = cursor.fetchone()

    cursor.close()

    if not row:
        return 0

    return row[0]

def user_favorite_list(conn, USER_SEQ, OFFSET):
    cursor = conn.cursor()

    sql = '''
        SELECT *
        FROM SAVED_QUERY
        WHERE USER_SEQ = :USER_SEQ
          AND IS_FAVORITE = 'Y'
        ORDER BY CREATED_AT DESC
        OFFSET :OFFSET ROWS FETCH NEXT 10 ROWS ONLY
    '''

    cursor.execute(sql, {
        "USER_SEQ": USER_SEQ,
        "OFFSET": OFFSET
    })

    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()

    cursor.close()

    return [dict(zip(columns, row)) for row in rows]

# 즐겨찾기 해제
def update_favorite_release(cursor, conn, ids):
    if not ids:
        return

    placeholders = []
    params = {}

    for i, v in enumerate(map(int, ids)):
        key = f"id{i}"
        placeholders.append(f":{key}")
        params[key] = v

    sql = f"""
    UPDATE SAVED_QUERY
    SET
        IS_FAVORITE = 'N',
        TITLE = NULL
    WHERE SAVED_QUERY_ID IN ({", ".join(placeholders)})
      AND IS_FAVORITE = 'Y'
    """

    cursor.execute(sql, params)
    conn.commit()

# 사용자 권한 관리 목록
def show_role_list(conn, ADMIN_ROLE, OFFSET):
    cursor = conn.cursor()

    sql = '''
        SELECT 
            USER_SEQ,
            USER_NAME, 
            DEPARTMENT, 
            FLOOR(SYSDATE - LOGIN_TIME) || '일 ' ||
            FLOOR(MOD((SYSDATE - LOGIN_TIME)*24, 24)) || '시간 전' AS TIME_DIFF,
            CASE 
                WHEN FLOOR(SYSDATE - LOGIN_TIME) >= 180
                THEN 'N'
                ELSE 'Y'
            END AS IS_ACTIVE
        FROM LOGIN_SESSION
        WHERE ADMIN_ROLE = :ADMIN_ROLE
        ORDER BY SYSDATE - LOGIN_TIME
        OFFSET :OFFSET ROWS FETCH NEXT 10 ROWS ONLY
    '''

    cursor.execute(sql, {
        "ADMIN_ROLE": ADMIN_ROLE,
        "OFFSET": OFFSET
    })

    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()

    cursor.close()

    return [dict(zip(columns, row)) for row in rows]

def total_role_list_len(conn, ADMIN_ROLE):
    cursor = conn.cursor()

    sql = """
        SELECT COUNT(1)
        FROM LOGIN_SESSION
        WHERE ADMIN_ROLE = :ADMIN_ROLE
    """

    cursor.execute(sql, {"ADMIN_ROLE": ADMIN_ROLE})
    row = cursor.fetchone()

    cursor.close()

    if not row:
        return 0

    return row[0]

def return_line_chart(df, x_axis, y_axis):
    df = df.sort_values(by=x_axis, ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_axis],
        y=df[y_axis],
        mode='lines',
        line=dict(
            color='#6A9C89',
            width=3
        ),
        hovertemplate=f'{x_axis}=%{{x}}<br>{y_axis}=%{{y:.2f}}<extra></extra>'
    ))    

    fig.update_layout(
        template="simple_white",
        height=400,
        margin=dict(l=100, r=40, t=20, b=40),

        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='#E5E7EB',
            linewidth=1,
            fixedrange=True,
            ticks="",
        ),

        yaxis=dict(
            showgrid=False,
            showline=False,
            fixedrange=True,
            ticks="",
        ),

        font=dict(
            family="Pretendard, -apple-system, BlinkMacSystemFont, sans-serif",
            size=14,
            color="#2c3e50"
        ),
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displayModeBar": False,
            "scrollZoom": False
        }
    )

def return_bar_chart(df, x_axis, y_axis):
    df = df.sort_values(by=x_axis, ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[x_axis],
        y=df[y_axis],
        orientation='h',
        marker=dict(
            color='#6A9C89',
            line=dict(width=0)
        ),
        # 🔥 숫자 제거
        hovertemplate=f'{y_axis}=%{{y}}<br>{x_axis}=%{{x:.2f}}<extra></extra>'
    ))
    
    fig.update_layout(
        template="simple_white",
        height=400,
        margin=dict(l=220, r=40, t=20, b=40),  # 🔥 왼쪽 여백 증가
        
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='#E5E7EB',
            linewidth=1,
            ticks="",
            fixedrange=True
        ),
        
        yaxis=dict(
            type='category',
            categoryorder='total ascending',
            showgrid=False,
            showline=False,
            ticks="",
            fixedrange=True
        ),
        
        font=dict(
            family="Pretendard, -apple-system, BlinkMacSystemFont, sans-serif",
            size=14,
            color="#2c3e50"
        ),
    )
    
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displayModeBar": False,
            "scrollZoom": False
        }
    )

def return_scatter_chart(df, x_axis, y_axis):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_axis],
        y=df[y_axis],
        mode='markers',
        marker=dict(
            size=8,
            color='#6A9C89',
            opacity=0.8
        ),
        hovertemplate=f'{x_axis}=%{{x}}<br>{y_axis}=%{{y:.2f}}<extra></extra>'
    ))

    fig.update_layout(
        template="simple_white",
        height=400,
        margin=dict(l=100, r=40, t=20, b=40),

        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='#E5E7EB',
            linewidth=1,
            fixedrange=True
        ),

        yaxis=dict(
            showgrid=False,
            showline=False,
            fixedrange=True
        ),

        font=dict(
            family="Pretendard, -apple-system, BlinkMacSystemFont, sans-serif",
            size=14,
            color="#2c3e50"
        ),
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displayModeBar": False,
            "scrollZoom": False
        }
    )

def return_query_line_chart(df, x_axis, y_axis):

    df = df[[x_axis, y_axis]].copy()
    df = df.sort_values(by=x_axis)

    full_hours = pd.DataFrame({x_axis: range(0, 25)})
    df_full = full_hours.merge(df, on=x_axis, how='left')
    df_full[y_axis] = df_full[y_axis].fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_full[x_axis],
        y=df_full[y_axis],
        mode='lines',
        line=dict(
            color="#1FA774",
            width=4,
            shape='spline'
        ),
        fill='tozeroy',
        fillcolor='rgba(31,167,116,0.2)'
    ))

    fig.update_layout(
        template="simple_white",
        height=150, # 이미지 비율에 맞춰 높이 조절
        width=750,
        # 아래쪽(b) 마진을 적절히 주어 시간 라벨이 잘리지 않게 함
        margin=dict(l=20, r=20, t=0, b=40),

        xaxis=dict(
            tickmode='array',
            tickvals=[0, 4, 8, 12, 16, 20, 24],
            ticktext=['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], # 끝부분 깔끔하게 비움
            showline=True,
            linecolor='#E5E7EB', # 연한 회색으로 축 선 표시
            linewidth=1,
            range=[0, 24],
            ticks="",           # 눈금만 제거
            showgrid=False,
            fixedrange=True
        ),

        yaxis=dict(
            # 최댓값이 작을 경우를 대비해 하단에 여유를 주되, 
            # 데이터가 있는 곳이 더 잘 보이도록 range 최적화
            range=[0, df_full[y_axis].max() * 1.1], 
            showticklabels=False,   # 라벨 제거
            ticks="",               # tick 표시 제거
            showline=False,         # 축 선 제거 (선도 없애려면)
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        font=dict(
            family="Pretendard, -apple-system, BlinkMacSystemFont, sans-serif",
            size=14,
            color="#2c3e50"
        ),
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',  # 👈 여기를 True로 변경
        config={
            "responsive": True,
            "displayModeBar": False,
            "scrollZoom": False
        }
    )

def all_query_count(conn):
    cursor = conn.cursor()

    sql = """
        SELECT *
        FROM ALL_QUERY_COUNT
        ORDER BY NOWTIME
    """
    cursor.execute(sql)

    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()

    cursor.close()

    if not rows:
        return None, 0, 0

    df = pd.DataFrame(rows, columns=columns)

    latest_query_count = df.iloc[-1]['QUERY_COUNT']
    latest_avg_query_time = df.iloc[-1]['AVG_CREATING_TIME']

    return df, latest_query_count, latest_avg_query_time

def execute_sql(sql, cur, limit=1000):
    safe_sql = sql.strip().rstrip(";")

    if "fetch first" not in safe_sql.lower():
        safe_sql = f"{safe_sql}\nFETCH FIRST {limit} ROWS ONLY"

    cur.execute(safe_sql)

    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(limit)

    return {
        "columns": cols,
        "rows": rows
    }

def return_top_5_query(cur):
    sql = '''select a.CHAT_SESSION_ID, b.saved_query_id, b.created_at, b.creating_time from CHAT_HISTORY a 
            join 
            (select * from saved_query
            where creating_time > 30) b 
            on a.question_id = b.question_id
            order by creating_time desc
            FETCH FIRST 5 ROWS ONLY
            '''

    cur.execute(sql)
    rows = cur.fetchall()
    return rows

# 해당 유저가 특정 테이블에 대한 권한을 가지고 있는지
def privilege_validation(cursor,user_name):
    sql = f'''SELECT TABLE_NAME
            FROM DBA_TAB_PRIVS
            WHERE GRANTEE IN (
                SELECT GRANTED_ROLE
                FROM DBA_ROLE_PRIVS
                WHERE GRANTEE = '{user_name}'
            )'''

    cursor.execute(sql)
    return [i[0] for i in cursor.fetchall()]

def load_llm_answer(conn, QUESTION_ID):
    cursor = conn.cursor()

    sql = """
        select ANSWER_TEXT from CHAT_HISTORY
        where QUESTION_ID = :QUESTION_ID
    """

    cursor.execute(sql, {"QUESTION_ID": QUESTION_ID})
    row = cursor.fetchone()

    cursor.close()

    if not row:
        return None

    return row[0]

def extract_tables(sql):
    parsed = parse_one(sql)
    tables = {table.name for table in parsed.find_all(Table)}
    return [i.upper() for i in list(tables)]

def select_to_csv_bytes_pandas(conn, sql, params, max_rows = 100000):
    if not sql.strip().lower().startswith("select"):
        raise ValueError("SELECT only")

    rows = 0
    chunks = []

    for chunk in pd.read_sql(sql, con=conn, params=params, chunksize=2000):
        chunks.append(chunk)
        rows += len(chunk)
        if rows >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True).head(max_rows)

    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.seek(0)
    return buf