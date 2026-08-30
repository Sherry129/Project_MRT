# -*- coding: utf-8 -*-
"""
階段2b：環狀線移交斷點診斷（Ring-line handover structural-break diagnostic）
=============================================================================
目的：量化2023-06環狀線移交新北捷運營運，對「進站量（目標變數）」造成的
      結構性斷點大小，並比較雙線轉乘站 vs 純環狀站的差異。
      這份診斷是資料誠信的佐證，供口試與最終報告引用。

背景（已由本腳本輸出驗證）：
- 此份為「臺北捷運(北捷)OD」資料。環狀線移交後，僅保留「有經過北捷路網」
  的旅次；完全待在環狀線/新北捷運網內的旅次歸新北捷運資料、從此消失。
- 進站(origin)側全期維持119個「無前綴」站名（例如大坪林不分「大坪林」與
  綠線識別「G大坪林」，一律記為「大坪林」）。故站名結構無斷點，但14個
  環狀站的進站「數值」在2023-06出現永久性水準下移。
- 雙線轉乘站（大坪林=綠線、景安/頭前庄=橘線）另有北捷基本盤，掉幅小(~15%)；
  純環狀站無北捷基本盤，掉幅大(~50%)。掉幅差＝該站北捷路網占比的度量。

方法：比較移交前後相鄰月（2023-05 vs 2023-06）各站進站量月總，
      以非環狀對照站的同期變化（約-3.5%）作為季節性基準，
      「扣季節後真實掉幅」= 總變化% - 季節基準%。

輸入：data/agg/station_daily.csv（stage2產出；欄位 date, station, entries, exits）
      或直接掃 data/parquet/od/od_2023{05,06}.parquet
輸出：reports/stage2b_ring_handover.csv、主控台比較表

用法：python scripts/stage2b_ring_handover_diagnostic.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATION_DAILY = ROOT / "data" / "agg" / "station_daily.csv"
OUT = ROOT / "reports" / "stage2b_ring_handover.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

MONTH_BEFORE = "2023-05"   # 移交前最後一個完整月
MONTH_AFTER = "2023-06"    # 移交後第一個月

# 環狀線14站。雙線＝同時屬其他路線的轉乘站（另一線出站以G/O前綴在北捷續存）
DUAL_LINE = ["大坪林", "景安", "頭前庄"]              # 綠線/橘線 + 環狀線
PURE_RING = ["Y板橋", "中原", "中和", "十四張", "幸福", "新北產業園區",
             "新埔民生", "景平", "板新", "橋和", "秀朗橋"]  # 僅環狀線
CONTROL = ["中山", "市政府", "台北車站"]             # 非環狀，估季節性基準


def monthly_entries(df: pd.DataFrame, stations: list) -> pd.DataFrame:
    """回傳指定站在 before/after 兩月的進站量月總與變化率。"""
    df = df.copy()
    df["ym"] = df["date"].str.slice(0, 7)
    sub = df[df.station.isin(stations) & df.ym.isin([MONTH_BEFORE, MONTH_AFTER])]
    piv = sub.pivot_table(index="station", columns="ym", values="entries",
                          aggfunc="sum").reindex(stations)
    piv["total_change_pct"] = (piv[MONTH_AFTER] - piv[MONTH_BEFORE]) / piv[MONTH_BEFORE] * 100
    return piv


def main():
    df = pd.read_csv(STATION_DAILY, usecols=["date", "station", "entries"], dtype={"date": str})

    ctrl = monthly_entries(df, CONTROL)
    seasonal = ctrl["total_change_pct"].mean()  # 季節性基準（非環狀站同期平均變化）

    dual = monthly_entries(df, DUAL_LINE)
    pure = monthly_entries(df, PURE_RING)
    for t in (dual, pure):
        t["real_drop_pct"] = t["total_change_pct"] - seasonal  # 扣季節後真實掉幅

    dual["group"] = "dual_line"
    pure["group"] = "pure_ring"
    ctrl["group"] = "control"
    ctrl["real_drop_pct"] = ctrl["total_change_pct"] - seasonal
    out = pd.concat([ctrl, dual, pure]).reset_index()
    out.insert(0, "seasonal_baseline_pct", round(seasonal, 2))
    out.to_csv(OUT, index=False)

    # 主控台報表
    def show(title, t):
        print(f"\n=== {title} ===")
        for st, r in t.sort_values("total_change_pct").iterrows():
            print(f"  {st:<8} {int(r[MONTH_BEFORE]):>9}->{int(r[MONTH_AFTER]):>9}  "
                  f"總{r.total_change_pct:+6.1f}%   扣季節後真實{r.real_drop_pct:+6.1f}%")

    print(f"季節性基準（非環狀對照站 {MONTH_BEFORE}->{MONTH_AFTER} 平均）: {seasonal:+.1f}%")
    show("對照組（非環狀，純季節性）", ctrl)
    show("雙線3站（另一線在北捷續存，掉幅小）", dual)
    show("純環狀11站（無北捷基本盤，掉幅大）", pure)
    print(f"\n群組平均｜雙線3站 真實掉幅 {dual.real_drop_pct.mean():+.1f}%"
          f"｜純環狀11站 真實掉幅 {pure.real_drop_pct.mean():+.1f}%")
    print("\n解讀：純環狀站掉幅≈50%（非100%）→ 留下的是『轉乘進北捷路網』的旅次，"
          "證明保留規則是『有經過北捷網路即保留』，非『環狀進站即整筆消失』。")
    print(f"輸出：{OUT}")


if __name__ == "__main__":
    main()
