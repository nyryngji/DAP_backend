import os
import io
import oracledb
import pandas as pd
from dotenv import load_dotenv

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