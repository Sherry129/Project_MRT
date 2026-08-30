"""
階段0：資料盤點腳本
====================
目的：在動手清理/建模前，先搞清楚兩份資料的實際樣貌，避免後面卡關。

支援兩種模式：
1. DuckDB 模式（推薦，速度快、記憶體佔用低，可直接對大型CSV下SQL查詢，不需要全部載入記憶體）
2. Pandas chunk 模式（備用，若環境沒有duckdb則用這個，逐塊讀取避免記憶體爆掉）

安裝（若尚未安裝）：
    pip install duckdb --break-system-packages

使用方式：
    python stage0_data_audit.py --file /path/to/od_data.csv --date_col 日期
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def audit_with_duckdb(filepath: str, date_col: str = "日期"):
    """
    用DuckDB直接對CSV下SQL查詢做盤點，不需要把整個檔案讀進記憶體。
    這是處理大型CSV（幾百MB到幾GB）最推薦的方式。
    """
    import duckdb

    con = duckdb.connect()  # in-memory連線
    # DuckDB可以直接對CSV檔案做SQL查詢，read_csv_auto會自動推斷欄位型態
    con.execute(f"CREATE VIEW raw_data AS SELECT * FROM read_csv_auto('{filepath}')")

    print(f"===== DuckDB盤點結果：{Path(filepath).name} =====\n")

    # 1. 總筆數與欄位
    schema = con.execute("DESCRIBE raw_data").fetchdf()
    print("欄位結構：")
    print(schema.to_string(index=False))

    total_rows = con.execute("SELECT COUNT(*) AS n FROM raw_data").fetchdf()["n"][0]
    print(f"\n總筆數：{total_rows:,}")

    # 2. 時間範圍
    if date_col in schema["column_name"].values:
        date_range = con.execute(
            f"SELECT MIN({date_col}) AS min_date, MAX({date_col}) AS max_date FROM raw_data"
        ).fetchdf()
        print(f"時間範圍：{date_range['min_date'][0]} ~ {date_range['max_date'][0]}")

        # 檢查資料是否連續（有無缺日）
        distinct_dates = con.execute(
            f"SELECT COUNT(DISTINCT {date_col}) AS n FROM raw_data"
        ).fetchdf()["n"][0]
        print(f"不重複日期數：{distinct_dates:,}")

    # 3. 各欄位缺失值統計
    print("\n缺失值統計：")
    for col in schema["column_name"]:
        na_count = con.execute(
            f'SELECT COUNT(*) AS n FROM raw_data WHERE "{col}" IS NULL'
        ).fetchdf()["n"][0]
        if na_count > 0:
            print(f"  {col}: {na_count:,} 筆缺失")

    # 4. 站名唯一值檢查（若有進站/出站欄位，抓常見命名）
    for col_candidate in ["進站", "出站", "站名", "station"]:
        if col_candidate in schema["column_name"].values:
            n_unique = con.execute(
                f'SELECT COUNT(DISTINCT "{col_candidate}") AS n FROM raw_data'
            ).fetchdf()["n"][0]
            print(f"\n{col_candidate} 欄位唯一值數量：{n_unique}")

    con.close()
    return {"total_rows": total_rows, "schema": schema}


def audit_with_pandas_chunks(filepath: str, date_col: str = "日期", chunksize: int = 500_000):
    """
    備用方案：若沒有安裝duckdb，用pandas分塊讀取的方式做盤點。
    不會一次把整個檔案讀進記憶體，而是逐塊累加統計量。
    """
    file_size_mb = os.path.getsize(filepath) / (1024**2)
    print(f"===== Pandas chunk盤點結果：{Path(filepath).name} =====")
    print(f"檔案大小：{file_size_mb:.1f} MB\n")

    total_rows = 0
    na_counts = None
    dtypes = None
    min_date, max_date = None, None

    reader = pd.read_csv(filepath, chunksize=chunksize, encoding="utf-8")
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        if dtypes is None:
            dtypes = chunk.dtypes
            na_counts = chunk.isna().sum()
        else:
            na_counts += chunk.isna().sum()

        if date_col in chunk.columns:
            chunk_dates = pd.to_datetime(chunk[date_col], errors="coerce")
            local_min, local_max = chunk_dates.min(), chunk_dates.max()
            min_date = local_min if min_date is None else min(min_date, local_min)
            max_date = local_max if max_date is None else max(max_date, local_max)

        if i % 20 == 0:
            print(f"已處理 {total_rows:,} 筆...")

    print(f"\n總筆數：{total_rows:,}")
    print(f"欄位型態：\n{dtypes}")
    print(f"\n缺失值統計：\n{na_counts[na_counts > 0]}")
    if min_date is not None:
        print(f"\n時間範圍：{min_date.date()} ~ {max_date.date()}")

    return {"total_rows": total_rows, "dtypes": dtypes, "na_counts": na_counts}


def main():
    parser = argparse.ArgumentParser(description="大型交通資料盤點工具")
    parser.add_argument("--file", required=True, help="CSV檔案路徑")
    parser.add_argument("--date_col", default="日期", help="日期欄位名稱")
    args = parser.parse_args()

    try:
        audit_with_duckdb(args.file, args.date_col)
    except ImportError:
        print("未安裝duckdb，改用pandas chunk模式（建議安裝duckdb以獲得更好效能：")
        print("pip install duckdb --break-system-packages）\n")
        audit_with_pandas_chunks(args.file, args.date_col)


if __name__ == "__main__":
    main()
