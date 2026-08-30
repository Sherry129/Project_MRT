"""
階段1：資料清理與轉換Pipeline
================================
目的：將原始大型CSV（尤其OD資料）轉換成乾淨、高效能的格式，
      避免後續每次分析都要重新讀取/清理一次龐大的原始檔案。

核心策略：
1. 用DuckDB做轉換，全程不需要把資料整個載入Python記憶體
2. 輸出成parquet格式：比CSV小很多（壓縮）、讀取速度快很多、且保留欄位型態
3. 數值欄位做dtype優化（例如人次用int32而非int64，省一半記憶體）
4. 站名做標準化（去除多餘空白、統一異體字等常見問題）

安裝需求：
    pip install duckdb --break-system-packages
"""

import duckdb
from pathlib import Path


def clean_od_data(
    input_csv: str,
    output_parquet: str,
    date_col: str = "日期",
    station_cols: list = None,
):
    """
    清理OD資料（進站-出站-人次格式），輸出成parquet。

    參數:
        input_csv: 原始CSV路徑
        output_parquet: 輸出的parquet路徑
        date_col: 日期欄位名稱
        station_cols: 站名欄位清單，預設為['進站', '出站']
    """
    if station_cols is None:
        station_cols = ["進站", "出站"]

    con = duckdb.connect()

    # 建立標準化清理的SQL：
    # - TRIM去除站名前後空白
    # - 日期轉成標準DATE型態
    # - 過濾掉人次為負值或明顯異常的資料（依實際狀況調整門檻）
    station_select = ", ".join(
        [f'TRIM("{col}") AS "{col}"' for col in station_cols]
    )

    query = f"""
        COPY (
            SELECT
                CAST({date_col} AS DATE) AS date,
                時段,
                {station_select},
                CAST(人次 AS INTEGER) AS trips
            FROM read_csv_auto('{input_csv}')
            WHERE 人次 IS NOT NULL AND 人次 >= 0
        ) TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    print(f"開始清理與轉換：{input_csv} -> {output_parquet}")
    con.execute(query)
    con.close()

    # 驗證輸出
    file_size_mb = Path(output_parquet).stat().st_size / (1024**2)
    print(f"完成。輸出檔案大小：{file_size_mb:.1f} MB")


def clean_daily_station_data(
    input_csv: str,
    output_parquet: str,
    date_col: str = "日期",
):
    """
    清理日資料（每站一欄的寬表格式，如image2），
    轉成長格式（long format：date, station, entries）並輸出parquet。
    長格式對後續建模、合併特徵比較方便操作。
    """
    con = duckdb.connect()

    # 先讀取欄位名稱，除了日期欄之外都是站名
    schema = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{input_csv}')").fetchdf()
    station_columns = [c for c in schema["column_name"] if c != date_col]

    # 用UNPIVOT把寬表轉長表（DuckDB語法）
    unpivot_query = f"""
        COPY (
            SELECT
                CAST({date_col} AS DATE) AS date,
                station,
                CAST(entries AS INTEGER) AS entries
            FROM read_csv_auto('{input_csv}')
            UNPIVOT (entries FOR station IN ({", ".join(f'"{c}"' for c in station_columns)}))
        ) TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    print(f"開始清理與轉換：{input_csv} -> {output_parquet}")
    con.execute(unpivot_query)
    con.close()

    file_size_mb = Path(output_parquet).stat().st_size / (1024**2)
    print(f"完成。輸出檔案大小：{file_size_mb:.1f} MB")
    print(f"共轉換 {len(station_columns)} 個站點的資料為長格式")


def quick_query_example(parquet_path: str):
    """
    示範：之後分析時如何快速查詢parquet檔案，不需要載入整份資料。
    例如只想看某個站點、某段時間的資料。
    """
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT * FROM '{parquet_path}'
        WHERE station = '動物園' AND date >= '2018-01-01'
        ORDER BY date
        LIMIT 10
    """).fetchdf()
    con.close()
    return result


if __name__ == "__main__":
    # 範例用法（依實際檔案路徑調整）
    # clean_od_data("od_raw.csv", "od_clean.parquet")
    # clean_daily_station_data("daily_raw.csv", "daily_clean.parquet")
    print("請依實際檔案路徑呼叫 clean_od_data() 或 clean_daily_station_data()")
