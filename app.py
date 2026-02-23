from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
import oracledb
from dotenv import load_dotenv
import os
from db_func import *
import uuid
import math
import json
import time
from llm import llm_answer, generate_medical_sql_explanation_json, extract_tables
import pandas as pd
import threading
from csv_to_file import *

app = Flask(__name__)
app.secret_key = 'super-secret-key'

load_dotenv('.env')

oracledb.init_oracle_client(lib_dir=r"C:\instant_client\instantclient_21_19")

pool = oracledb.create_pool(
    user=os.getenv('user'),
    password=os.getenv('password'),
    dsn=os.getenv('ORACLE_DSN'),
    min=2,        # 최소 커넥션
    max=10,       # 최대 커넥션
    increment=1,
)

conn = pool.acquire()
cur = conn.cursor()

dsn = os.getenv("ORACLE_DSN")

role_pool = None

doctor_pool = oracledb.create_pool(
    user="TEAM8_DOCTOR_USER",
    password="Oracle_4U!!12",
    dsn=dsn,
    min=2,
    max=5,
)

research_pool = oracledb.create_pool(
    user="TEAM8_RESEARCH_USER",
    password="Oracle_4U!!12",
    dsn=dsn,
    min=2,
    max=5,
)

admin_pool = oracledb.create_pool(
    user="TEAM8_ADMIN_USER",
    password="Oracle_4U!!12",
    dsn=dsn,
    min=1,
    max=3,
)

system_pool = oracledb.create_pool(
    user="TEAM8_SYSTEM_USER",
    password="Oracle_4U!!12",
    dsn=dsn,
    min=1,
    max=3,
)

def get_pool_by_role(role):
    if role == "CLINICAL":
        return "TEAM8_DOCTOR_USER", doctor_pool
    
    elif role == "RESEARCHER":
        return "TEAM8_RESEARCH_USER", research_pool
    
    elif role == "ADMINISTRATION":
        return "TEAM8_ADMIN_USER", admin_pool
    
    elif role == "SYSTEM":
        return "TEAM8_SYSTEM_USER", system_pool
    else:
        raise Exception("Invalid role")

def generate_answer(human_question, answer_id, USER_SEQ, chat_session_id, user_role):

    role_pool = None
    role_conn = None
    role_cursor = None

    try:
        user_name, role_pool = get_pool_by_role(user_role)
        role_conn = role_pool.acquire()
        role_cursor = role_conn.cursor()

        start_time = time.perf_counter()

        # ✅ LLM 호출
        sql, sql_json, bind_query, bind_dict = llm_answer(
            human_question, cur
        )

        print(f'현재 생성된 쿼리 : {bind_query, bind_dict}')

        # 🔥 권한 체크 (하나라도 없으면 즉시 종료)
        selected_tables = extract_tables(sql)

        for table_name in selected_tables:
            if privilege_validation(cur, table_name, user_name) == 'NO_PRIVILEGE':
                
                update_llm_answer(
                    cur,
                    conn,
                    "해당 테이블에 대한 권한이 없습니다.",
                    answer_id
                )
                conn.commit()
                return   # 🚨 즉시 종료

        print('권한 체크 종료')

        bind_json = json.dumps(bind_dict, ensure_ascii=False)
        elapsed_seconds = int(time.perf_counter() - start_time)

        # ✅ SQL 정보 저장
        save_sql(
            cur,
            conn,
            USER_SEQ,
            chat_session_id,
            answer_id,
            bind_query,
            bind_json,
            elapsed_seconds
        )

        llm_natural_answer = generate_medical_sql_explanation_json(
            human_question,
            sql_json,
            execution_summary=None
        )
        
        # ✅ 정상 응답
        update_llm_answer(
            cur,
            conn,
            f"{llm_natural_answer}",
            answer_id
        )

        conn.commit()

    except Exception as e:
        print("generate_answer error:", e)
        if conn:
            conn.rollback()

        if cur and conn:
            update_llm_answer(
                cur,
                conn,
                "❌ 답변 생성 중 오류가 발생했습니다.",
                answer_id
            )
            conn.commit()

    finally:
        if role_cursor:
            role_cursor.close()
        if role_conn:
            role_conn.close()
        if cur:
            cur.close()
        if conn:
            conn.close()


# 로그인 화면 1
@app.route('/', methods=['GET'])
def view_login_page(): 
   return render_template('login.html')

# 로그인 화면 2
@app.route('/', methods=['POST'])
def login_func():
    user_id = request.form['user_id']
    user_password = request.form['user_password']

    # 🔹 로그인 검증은 공용 conn 사용
    user_info = login_with_db(conn, user_id, user_password)

    if user_info is not None:
        if int(user_info['LAST_LOGIN_TIME']) >= 180:
            flash('장기간 미접속으로 인해 계정이 잠겼습니다.\n관리자에게 문의하세요.')
            return render_template('login.html')
    
        else:
            session['user_info'] = user_info
            USER_SEQ = user_info['USER_SEQ']
            user_role = user_info['ADMIN_ROLE']   # 🔥 여기서 role 가져옴

            session['db_role'] = user_role

            if user_role != 'SYSTEM':
                chat_session_id = f"{USER_SEQ}_{uuid.uuid4()}"
                save_chat_session(cur, conn, chat_session_id, USER_SEQ)
                session['chat_session_id'] = chat_session_id
                return redirect(url_for('view_already_chat_page', chat_session_id=chat_session_id))
            else:
                return redirect(url_for('view_system_log_page'))

    else:
        flash('아이디나 비밀번호가 일치하지 않습니다.')
        return render_template('login.html')

# 로그아웃 화면
@app.route('/logout')
def logout():
    session.clear() 
    return redirect(url_for('view_login_page'))

# 새로운 채팅 
@app.route('/new-chat')
def new_chat():
    user_info = session.get('user_info')
    if not user_info:
        return redirect(url_for('view_login_page'))

    USER_SEQ = user_info['USER_SEQ']
    chat_session_id = f"{USER_SEQ}_{uuid.uuid4()}"

    save_chat_session(cur, conn, chat_session_id, USER_SEQ)
    session['chat_session_id'] = chat_session_id

    return redirect(url_for('view_already_chat_page', chat_session_id=chat_session_id))

@app.route('/main/<chat_session_id>', methods=['GET', 'POST'])
def view_already_chat_page(chat_session_id):

    user_info = session.get('user_info')
    user_role = user_info['ADMIN_ROLE']

    if not user_info:
        return redirect(url_for('view_login_page'))

    USER_SEQ = user_info['USER_SEQ']
    session['chat_session_id'] = chat_session_id
    session['llm_answer_id'] = None

    # -----------------------------
    # POST (질문 입력)
    # -----------------------------
    if request.method == "POST":
        human_question = request.form.get('human_question')

        if human_question:
            conn = pool.acquire()
            cur = conn.cursor()

            try:
                save_user_question(cur, conn, chat_session_id, human_question)

                answer_id = save_llm_answer(
                    cur,
                    conn,
                    chat_session_id,
                    "⏳ 생성 중입니다..."
                )

                conn.commit()

            finally:
                cur.close()
                conn.close()

            threading.Thread(
                target=generate_answer,
                args=(human_question, answer_id, USER_SEQ, chat_session_id, user_role),
                daemon=True
            ).start()

            session['pending_answer_id'] = answer_id

        return redirect(url_for('view_already_chat_page', chat_session_id=chat_session_id))

    # -----------------------------
    # GET (채팅 로딩)
    # -----------------------------
    conn = pool.acquire()

    try:
        chat_history = load_chat_history(conn, chat_session_id)
        user_chat_first_dialog = user_chat_list(conn, USER_SEQ)

    finally:
        conn.close()

    chat_blocks = []
    current_block = None

    for row in chat_history:
        if row["QUESTION_TEXT"]:
            if current_block:
                chat_blocks.append(current_block)

            current_block = {
                "question": row["QUESTION_TEXT"],
                "answer": None,
                "table": None,
                "answer_id": None
            }

        elif row["ANSWER_TEXT"] and current_block:
            current_block["answer"] = row["ANSWER_TEXT"]
            current_block["answer_id"] = row["QUESTION_ID"]

    if current_block:
        chat_blocks.append(current_block)

    # ------------------------------------
    # 🔥 최근 답변 5개만 테이블 실행
    # ------------------------------------
    MAX_PREVIEW = 5
    preview_count = 0

    conn = pool.acquire()
    try:
        # 최신 답변부터 처리 (뒤에서부터)
        for block in reversed(chat_blocks):
            if preview_count >= MAX_PREVIEW:
                break

            if block["answer"] and not block["answer"].startswith("⏳"):
                sql_info = load_sql_for_table_info(conn, block["answer_id"])

                if sql_info:
                    bind_values = json.loads(sql_info["BIND_VALUES"]) if sql_info.get("BIND_VALUES") else {}
                    table_info = make_table_from_sql(conn, sql_info["SQL_TEXT"], bind_values, limit=5)
                    block["table"] = table_info
                
    finally:
        conn.close()

    pending_answer_id = session.pop('pending_answer_id', None)

    return render_template(
        'main.html',
        user_info=user_info,
        chat_blocks=chat_blocks,
        user_chat_list=user_chat_first_dialog,
        pending_answer_id=pending_answer_id
    )

@app.route('/check_answer_status/<int:answer_id>')
def check_answer_status(answer_id):
    try:
        conn = pool.acquire()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ANSWER_TEXT
            FROM CHAT_HISTORY
            WHERE QUESTION_ID = :answer_id
        """, {"answer_id": answer_id})

        row = cursor.fetchone()

        cursor.close()
        conn.close()

        if not row:
            return jsonify({'status': 'error'}), 404

        answer_text = row[0] or ""

        # 🔥 placeholder 기준으로 판단
        if answer_text.startswith("⏳"):
            return jsonify({'status': 'pending'})

        return jsonify({
            'status': 'complete',
            'answer': answer_text
        })

    except Exception as e:
        print("check_answer_status error:", e)
        return jsonify({'status': 'error'}), 500
    
@app.route("/popup/<int:answer_id>", methods=['GET', 'POST'])
def popup(answer_id):

    formatted_sql = None
    favorite_title = None
    update_favorite_title = None
    graph_html = None
    graph_error = None  # 👈 session 대신 변수로
    table_info = []
    bind_values = {}

    # 1️⃣ SQL 로드
    raw_sql = load_sql_for_table_info(conn, answer_id)

    if raw_sql:
        original_sql = raw_sql['SQL_TEXT']
        formatted_sql = sql_parsing(original_sql)

        if raw_sql.get("BIND_VALUES"):
            bind_values = json.loads(raw_sql["BIND_VALUES"])
    else:
        return "SQL 정보를 찾을 수 없습니다.", 404

    # 2️⃣ SQL 실행
    try:
        table_info = make_table_from_sql(conn, original_sql, bind_values, limit=20)
    except Exception as e:
        return f"SQL 실행 실패: {str(e)}", 500

    df = pd.DataFrame(table_info['rows'], columns=table_info['columns'])

    # 3️⃣ 즐겨찾기 조회
    favorite_result = isit_favorite(conn, answer_id)

    if favorite_result:
        update_favorite_title = favorite_result['IS_FAVORITE']
        favorite_title = favorite_result['TITLE']

    # 4️⃣ POST 처리
    if request.method == 'POST':

        favorite_title = request.form.get('favorite_title')
        x_axis = request.form.get("x_axis")
        y_axis = request.form.get("y_axis")
        graph_type = request.form.get("graph_type")

        # 즐겨찾기 업데이트
        if favorite_title is not None:
            update_favorite(cur, conn, favorite_title, answer_id)

        # 그래프 생성
        if x_axis and y_axis and graph_type:
            
            if x_axis not in df.columns or y_axis not in df.columns:
                graph_error = "선택한 축이 데이터에 존재하지 않습니다."
            elif graph_type in ["line", "scatter"] and not pd.api.types.is_numeric_dtype(df[y_axis]):
                graph_error = "Y축은 숫자형이어야 합니다."
            else:
                try:
                    if graph_type == "line":
                        graph_html = return_line_chart(df, x_axis, y_axis)
                    elif graph_type == "bar":
                        graph_html = return_bar_chart(df, x_axis, y_axis)
                    elif graph_type == "scatter":
                        graph_html = return_scatter_chart(df, x_axis, y_axis)
                    else:
                        graph_error = "지원하지 않는 그래프 타입입니다."

                except Exception as e:
                    graph_error = f"그래프 생성 중 오류: {str(e)}"
    
    return render_template(
        "popup.html",
        answer_id=answer_id,
        sql=formatted_sql,
        table_info=table_info,
        favorite_title=favorite_title,
        update_favorite_title=update_favorite_title,
        favorite_result=favorite_result,
        graph_html=graph_html,
        graph_error=graph_error
    )

@app.route("/popup/<int:answer_id>/download")
def popup_download(answer_id):

    favorite_title = None

    favorite_result = isit_favorite(conn, answer_id)

    if favorite_result:
        favorite_title = favorite_result['TITLE']

    raw_sql = load_sql_for_table_info(conn, answer_id)

    if not raw_sql:
        return "SQL 정보를 찾을 수 없습니다.", 404

    original_sql = raw_sql['SQL_TEXT']
    bind_values = {}

    if raw_sql.get("BIND_VALUES"):
        bind_values = json.loads(raw_sql["BIND_VALUES"])

    csv_buf = select_to_csv_bytes_pandas(conn,
        original_sql,
        params=bind_values,
        max_rows=100000
    )

    return send_file(
        csv_buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{favorite_title}.csv"
    )

# 즐겨찾기
@app.route('/favorite', methods=['GET', 'POST'])
def view_favorite_page():
    user_info = session.get('user_info')
    USER_SEQ = user_info['USER_SEQ']

    session['is_favorite_clicked'] = True
    user_chat_first_dialog = user_chat_list(conn, USER_SEQ)

    page = int(request.args.get('page', 1))
    offset = (page - 1) * 10
    print(USER_SEQ)
    fav_list = user_favorite_list(conn, USER_SEQ, offset)

    total_list = max(total_favorite_list_len(conn, USER_SEQ),0)
    total_pages = 1
    print("page:", page)

    if total_list > 0:
        total_pages = math.ceil(total_list / 10)

    if request.method == 'POST':
        for_delete = request.form.getlist('fav_ids')
        print(for_delete)
        if len(for_delete) > 0:
            update_favorite_release(cur, conn, for_delete)
            return redirect(url_for('view_favorite_page'))

    return render_template('main.html', is_favorite = session['is_favorite_clicked'], user_info=user_info,
                           user_chat_list = user_chat_first_dialog, fav_list = fav_list, page = page, total_pages = total_pages, total_list = total_list)

@app.route('/system_log', methods=['GET', 'POST'])
def view_system_log_page():
    user_info = session.get('user_info')
    session['system_log'] = True

    all_query_info, latest_query_count, latest_avg_query_time = all_query_count(conn)
    cpu_use = 0

    avg_query_time_per = all_query_info.iloc[-2]['AVG_CREATING_TIME'] - all_query_info.iloc[-1]['AVG_CREATING_TIME']
    query_count_per = round(all_query_info.iloc[-1]['TIME_QUERY'] / all_query_info.iloc[-1]['QUERY_COUNT'] * 100,2)

    cpu_use = round((all_query_info.iloc[-1]['CPU_CS'] / 360000) * 100, 2)

    # CPU 사용률
    if len(all_query_info) == 1:
        cpu_use_diff = round((all_query_info.iloc[-1]['CPU_CS'] - 0) / 360000, 2) * 100
    else:
        cpu_use_diff = round((all_query_info.iloc[-1]['CPU_CS'] - all_query_info.iloc[-2]['CPU_CS']) / 360000, 2) * 100

    query_graph_html = return_query_line_chart(all_query_info, 'NOWTIME', 'TIME_QUERY')

    top_5_query_rows = return_top_5_query(cur)

    slow_query_count = all_query_info['OVER_30S'].iloc[-1]
    slow_query_diff = all_query_info['OVER_30S'].iloc[-1] - all_query_info['OVER_30S'].iloc[-2]

    return render_template('main.html', user_info=user_info, 
                           system_loc = session['system_log'], 
                           latest_query_count = latest_query_count, 
                           query_count_per = query_count_per,
                           latest_avg_query_time = latest_avg_query_time,
                           cpu_use = cpu_use,
                           cpu_use_diff = cpu_use_diff,
                           query_graph_html = query_graph_html,
                           top_5_query_rows = top_5_query_rows,
                           avg_query_time_per = avg_query_time_per,
                           slow_query_count = slow_query_count,
                           slow_query_diff = slow_query_diff)


@app.route('/manage_role', methods=['GET', 'POST'])
def view_manage_role_page():
    user_info = session.get('user_info')
    session['system_log'] = False
    role = request.form.get("role") or request.args.get("role") or "CLINICAL"
    print(role)
    page = int(request.args.get('page', 1))
    offset = (page - 1) * 10
    selected_role_user_list = show_role_list(conn, role, offset)

    total_list = max(total_role_list_len(conn, role),0)
    total_pages = 1

    if total_list > 0:
        total_pages = math.ceil(total_list / 10)

    if request.is_json:
        data = request.get_json()
        if data['is_active'] == 'N':
            print(data) # 나중에 grant로 권한 바꿔야 함
    
    return render_template('main.html', user_info=user_info, system_loc = session['system_log'], 
                           selected_role_user_list = selected_role_user_list,
                           total_list = total_list, total_pages = total_pages,
                           page = page, role = role
                           )

if __name__ == '__main__':  
   app.run(debug=True, use_reloader=True)