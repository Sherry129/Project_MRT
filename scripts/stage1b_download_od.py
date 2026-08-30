"""
階段1b：OD月檔批次下載 + 轉parquet pipeline
==============================================
資料來源：臺北捷運各站分時進出量統計(OD)，臺北市資料大平臺
  https://data.taipei/dataset/detail?id=63f31c7e-7fc3-418b-bd82-b95158755b4d
月檔URL格式（2017-01 ~ 2026-05皆有效，已於2026-07驗證）：
  http://tcgmetro.blob.core.windows.net/stationod/臺北捷運每日分時各站OD流量統計資料_YYYYMM.csv

行為：
1. 依序處理 START_MONTH ~ END_MONTH 每個月份
2. 若該月parquet已存在 → 跳過（可重複執行、中斷後續跑）
3. 若 data/raw/ 已有該月CSV（含手動下載的 OD_YYYYMM.csv 命名）→ 直接轉檔不下載
4. 下載（串流寫入.tmp檔，成功才改名；失敗重試3次）
5. 用DuckDB轉parquet（ZSTD壓縮、站名TRIM、人次>=0過濾、int32）
6. 驗證parquet：列數>0且日期範圍落在該月內 → 通過才刪除原始CSV

用法（在專案根目錄執行）：
    python scripts/stage1b_download_od.py

安裝需求：
    pip install duckdb --break-system-packages
"""

import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import duckdb

# ── 設定 ────────────────────────────────────────────────
START_MONTH = (2023, 1)
END_MONTH = (2026, 5)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet" / "od"
BASE_URL = "http://tcgmetro.blob.core.windows.net/stationod/"
FILENAME_TPL = "臺北捷運每日分時各站OD流量統計資料_{ym}.csv"
DELETE_CSV_AFTER_CONVERT = True
RETRIES = 3
MIN_EXPECTED_ROWS = 1_000_000  # 每月正常約800萬列，低於此值視為異常


def month_range(start: tuple, end: tuple):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def find_existing_csv(ym: str) -> Path | None:
    """接受官方命名與手動命名（如 OD_202605.csv）兩種格式。"""
    for name in (FILENAME_TPL.format(ym=ym), f"OD_{ym}.csv"):
        p = RAW_DIR / name
        if p.exists():
            return p
    return None


def download(ym: str) -> Path:
    url = BASE_URL + urllib.parse.quote(FILENAME_TPL.format(ym=ym))
    dest = RAW_DIR / FILENAME_TPL.format(ym=ym)
    tmp = dest.with_suffix(".csv.tmp")
    for attempt in range(1, RETRIES + 1):
        try:
            print(f"  下載中（第{attempt}次嘗試）...")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            print(f"  下載完成：{dest.name}（{dest.stat().st_size / 1e6:.0f} MB）")
            return dest
        except Exception as e:
            print(f"  失敗：{e}")
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(5 * attempt)
    raise RuntimeError(f"{ym} 下載失敗，已重試{RETRIES}次")


def convert_and_verify(csv_path: Path, parquet_path: Path, year: int, month: int):
    con = duckdb.connect()
    tmp_pq = parquet_path.with_suffix(".parquet.tmp")
    con.execute(f"""
        COPY (
            SELECT
                CAST(日期 AS DATE)      AS date,
                CAST(時段 AS TINYINT)   AS hour,
                TRIM(進站)              AS origin,
                TRIM(出站)              AS destination,
                CAST(人次 AS INTEGER)   AS trips
            FROM read_csv_auto('{csv_path.as_posix()}')
            WHERE 人次 IS NOT NULL AND 人次 >= 0
        ) TO '{tmp_pq.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n, dmin, dmax = con.execute(
        f"SELECT COUNT(*), MIN(date), MAX(date) FROM '{tmp_pq.as_posix()}'"
    ).fetchone()
    con.close()

    month_start = date(year, month, 1)
    month_end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    if n < MIN_EXPECTED_ROWS:
        tmp_pq.unlink()
        raise RuntimeError(f"驗證失敗：僅{n}列（門檻{MIN_EXPECTED_ROWS}），保留CSV待查")
    # 註：12月檔含跨年夜通宵班次，日期會延伸到隔年1/1，屬正常資料，故允許 dmax == month_end
    if not (month_start <= dmin and dmax <= month_end):
        tmp_pq.unlink()
        raise RuntimeError(f"驗證失敗：日期範圍{dmin}~{dmax}超出該月，保留CSV待查")

    tmp_pq.rename(parquet_path)
    print(f"  轉檔完成：{parquet_path.name}"
          f"（{n:,}列，{parquet_path.stat().st_size / 1e6:.0f} MB，{dmin}~{dmax}）")
    if DELETE_CSV_AFTER_CONVERT:
        csv_path.unlink()
        print("  已刪除原始CSV")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    months = list(month_range(START_MONTH, END_MONTH))
    failed = []

    for i, (y, m) in enumerate(months, 1):
        ym = f"{y}{m:02d}"
        parquet_path = PARQUET_DIR / f"od_{ym}.parquet"
        print(f"[{i}/{len(months)}] {ym}")
        if parquet_path.exists():
            print("  parquet已存在，跳過")
            continue
        try:
            csv_path = find_existing_csv(ym) or download(ym)
            convert_and_verify(csv_path, parquet_path, y, m)
        except Exception as e:
            print(f"  ✗ {e}")
            failed.append(ym)

    print("\n" + "=" * 40)
    print(f"完成 {len(months) - len(failed)}/{len(months)} 個月份")
    if failed:
        print(f"失敗月份（重跑本腳本即可續傳）：{', '.join(failed)}")


if __name__ == "__main__":
    main()
