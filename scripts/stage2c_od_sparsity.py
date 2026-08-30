# -*- coding: utf-8 -*-
"""
階段2c：OD 零流量（稀疏性）診斷
================================
輸入：data/parquet/od/od_YYYYMM.parquet（stage1b產出）
輸出：data/agg/od_sparsity.csv

用途
----
報告 §3.3 主張「分時 OD 有相當比例為零流量，站對層級則幾乎不為零」，並以此作為
「不做 OD 級預測、改以站級日進站量為目標」的依據。這個主張原本只寫在敘述裡、
沒有腳本產出，本檔補上可重現的計算。

四種口徑（分母不同，數字差很多，引用時務必指明是哪一種）
--------------------------------------------------------
A. 原始紀錄層級  ：資料檔實際存在的列中，trips = 0 的比例。
                   北捷開放資料會把當日「有列出但沒人搭」的組合以 0 寫出來，
                   所以這個比例直接反映日層級分時 OD 的稀疏程度。
B. 完整格點層級  ：站對 × 時段 × 日 的理論格點（119² × H × 1,247）中，
                   沒有任何流量的比例（含資料檔根本沒列出的組合）。
C. 站對×時段累計 ：把 1,247 天加總後，站對 × 時段（119² × H）中全期皆為 0 的比例。
D. 站對累計      ：把 1,247 天與所有時段都加總後，站對（119²）中全期皆為 0 的比例。
                   這就是報告所說的「不分時段的站對層級零比例」。

H 同時提供兩版：H=24（全部時段）與 H=21（排除凌晨 02–04 時，該時段正常營運無班次，
只有跨年／白晝之夜等延長營運日才有流量，見 stage2_eda.py 的說明）。

站名處理與 stage2_eda.py 一致：202305 過渡月的 G/O 前綴變體先正規化，
確保 origin 與 destination 兩側都是同一組 119 個站名。

用法：python scripts/stage2c_od_sparsity.py
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PQ = ROOT / "data" / "parquet" / "od"
AGG = ROOT / "data" / "agg"
AGG.mkdir(parents=True, exist_ok=True)

# 與 stage2_eda.py / stage4_features.py 相同的站名正規化
FIX = {"G大坪林": "大坪林", "O景安": "景安", "O頭前庄": "頭前庄"}
NIGHT_HOURS = (2, 3, 4)   # 正常營運無班次的凌晨時段


def main():
    con = duckdb.connect()
    src = f"read_parquet('{(PQ / 'od_*.parquet').as_posix()}')"
    fix_sql = "CASE destination " + " ".join(
        f"WHEN '{k}' THEN '{v}'" for k, v in FIX.items()) + " ELSE destination END"
    con.execute(f"""
        CREATE VIEW od AS
        SELECT date, hour, origin, {fix_sql} AS destination, trips FROM {src}
    """)

    n_days = con.execute("SELECT COUNT(DISTINCT date) FROM od").fetchone()[0]
    n_st = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT origin AS s FROM od UNION SELECT destination FROM od
        )""").fetchone()[0]
    n_pairs = n_st * n_st
    print(f"站數 {n_st}｜站對格數 {n_pairs:,}｜天數 {n_days:,}")

    rows = []

    # ---- A. 原始紀錄層級 ----
    for hours, tag in ((24, "H=24（全部時段）"), (21, "H=21（排除02-04時）")):
        cond = "TRUE" if hours == 24 else f"hour NOT IN {NIGHT_HOURS}"
        n_rows, n_zero = con.execute(
            f"SELECT COUNT(*), SUM(CASE WHEN trips = 0 THEN 1 ELSE 0 END) FROM od WHERE {cond}"
        ).fetchone()
        rows.append(dict(scope="A_原始紀錄（站對×時段×日，資料檔實際列）", hours=hours,
                         denominator=n_rows, zero_cells=n_zero,
                         zero_pct=round(100 * n_zero / n_rows, 2), note=tag))

    # ---- B/C/D. 以格點為分母 ----
    for hours in (24, 21):
        cond = "TRUE" if hours == 24 else f"hour NOT IN {NIGHT_HOURS}"

        # B：站對 × 時段 × 日
        nz = con.execute(
            f"SELECT COUNT(*) FROM (SELECT date, hour, origin, destination FROM od "
            f"WHERE trips > 0 AND {cond} GROUP BY 1,2,3,4)").fetchone()[0]
        grid = n_pairs * hours * n_days
        rows.append(dict(scope="B_完整格點（站對×時段×日）", hours=hours,
                         denominator=grid, zero_cells=grid - nz,
                         zero_pct=round(100 * (grid - nz) / grid, 2),
                         note="含資料檔未列出的組合"))

        # C：站對 × 時段（全期加總）
        nz = con.execute(
            f"SELECT COUNT(*) FROM (SELECT hour, origin, destination FROM od "
            f"WHERE trips > 0 AND {cond} GROUP BY 1,2,3)").fetchone()[0]
        grid = n_pairs * hours
        rows.append(dict(scope="C_站對×時段（全期加總）", hours=hours,
                         denominator=grid, zero_cells=grid - nz,
                         zero_pct=round(100 * (grid - nz) / grid, 2),
                         note="全期皆為0才算零"))

    # D：站對（全期、不分時段加總）
    nz = con.execute(
        "SELECT COUNT(*) FROM (SELECT origin, destination FROM od "
        "WHERE trips > 0 GROUP BY 1,2)").fetchone()[0]
    rows.append(dict(scope="D_站對（全期、不分時段加總）", hours=None,
                     denominator=n_pairs, zero_cells=n_pairs - nz,
                     zero_pct=round(100 * (n_pairs - nz) / n_pairs, 2),
                     note="報告所稱「站對層級零比例」"))

    con.close()

    out = (pd.DataFrame(rows)[["scope", "hours", "denominator", "zero_cells", "zero_pct", "note"]]
           .sort_values(["scope", "hours"], na_position="last").reset_index(drop=True))
    out.to_csv(AGG / "od_sparsity.csv", index=False)
    print("\n=== OD 零流量診斷 ===")
    print(out.to_string(index=False))
    a24 = out.query("scope.str.startswith('A_') and hours == 24").zero_pct.iloc[0]
    d = out.query("scope.str.startswith('D_')").zero_pct.iloc[0]
    print(f"\n報告可引用：分時 OD（站對×時段×日）約 {a24:.0f}% 的紀錄為零流量；"
          f"站對層級（全期加總）零比例為 {d:.2f}%。")
    print("已輸出", AGG / "od_sparsity.csv")


if __name__ == "__main__":
    main()
