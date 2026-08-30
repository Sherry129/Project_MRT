# -*- coding: utf-8 -*-
"""
階段4：特徵工程 — 建模用feature table
================================
輸入：data/parquet/od/od_*.parquet、data/agg/station_daily.csv（stage2產出）
輸出：data/features/calendar_daily.csv、station_static.csv、feature_table.csv

設計原則（詳見 data/features/feature_dictionary.md）：
- 目標變數 = 每站每日進站量（環狀線出站斷點不影響）
- lag/rolling特徵一律shift(1)防當日洩漏
- 站點靜態特徵（分群/PageRank）僅用訓練期2023-2024計算
- 假日/補班：政府辦公日曆表；颱風停班：北市官方紀錄（皆為事前可得資訊）

安裝需求：pip install duckdb pandas numpy
用法：python scripts/stage4_features.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PQ = (ROOT / "data" / "parquet" / "od" / "od_*.parquet").as_posix()
AGG = ROOT / "data" / "agg"
FEAT = ROOT / "data" / "features"
FEAT.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2024-12-31"  # 靜態特徵計算窗（防洩漏）
FIX = {"G大坪林": "大坪林", "O景安": "景安", "O頭前庄": "頭前庄"}
YLINE = ["Y板橋", "中原", "中和", "十四張", "大坪林", "幸福", "新北產業園區",
         "新埔民生", "景安", "景平", "板新", "橋和", "秀朗橋", "頭前庄"]

# ── 日曆特徵（來源：政府行政機關辦公日曆表 data.gov.tw 14718；
#    台北市停班：臺北市歷次天然災害停止上班上課訊息）──
WEEKDAY_HOLIDAYS = {
    "2023-01-02": "補假", "2023-01-20": "小年夜", "2023-01-23": "春節", "2023-01-24": "春節",
    "2023-01-25": "補假", "2023-01-26": "補假", "2023-01-27": "調整放假", "2023-02-27": "調整放假",
    "2023-02-28": "和平紀念日", "2023-04-03": "調整放假", "2023-04-04": "兒童節", "2023-04-05": "民族掃墓節",
    "2023-06-22": "端午節", "2023-06-23": "調整放假", "2023-09-29": "中秋節", "2023-10-09": "調整放假",
    "2023-10-10": "國慶日",
    "2024-01-01": "開國紀念日", "2024-02-08": "小年夜", "2024-02-09": "農曆除夕", "2024-02-12": "春節",
    "2024-02-13": "補假", "2024-02-14": "補假", "2024-02-28": "和平紀念日", "2024-04-04": "兒童節及民族掃墓節",
    "2024-04-05": "補假", "2024-06-10": "端午節", "2024-09-17": "中秋節", "2024-10-10": "國慶日",
    "2025-01-01": "開國紀念日", "2025-01-27": "小年夜", "2025-01-28": "農曆除夕", "2025-01-29": "春節",
    "2025-01-30": "春節", "2025-01-31": "春節", "2025-02-28": "和平紀念日", "2025-04-03": "補假",
    "2025-04-04": "兒童節及民族掃墓節", "2025-05-30": "補假", "2025-09-29": "補假", "2025-10-06": "中秋節",
    "2025-10-10": "國慶日", "2025-10-24": "補假", "2025-12-25": "行憲紀念日",
    "2026-01-01": "開國紀念日", "2026-02-16": "農曆除夕", "2026-02-17": "春節", "2026-02-18": "春節",
    "2026-02-19": "春節", "2026-02-20": "補假", "2026-02-27": "補假", "2026-04-03": "補假",
    "2026-04-06": "補假", "2026-05-01": "勞動節",
}
WEEKEND_NAMES = {
    "2023-01-01": "開國紀念日", "2023-01-21": "農曆除夕", "2023-01-22": "春節",
    "2024-02-10": "春節", "2024-02-11": "春節", "2025-05-31": "端午節", "2025-09-28": "孔子誕辰紀念日",
    "2025-10-25": "臺灣光復節", "2026-02-15": "小年夜", "2026-02-28": "和平紀念日",
    "2026-04-04": "兒童節", "2026-04-05": "清明節",
}
MAKEUP_WORKDAYS = {"2023-01-07", "2023-02-04", "2023-02-18", "2023-03-25", "2023-06-17",
                   "2023-09-23", "2024-02-17", "2025-02-08"}
LNY_SPANS = [("2023-01-21", "2023-01-24"), ("2024-02-09", "2024-02-12"),
             ("2025-01-28", "2025-01-31"), ("2026-02-16", "2026-02-19")]
TYPHOON = {"2023-08-03": "卡努", "2024-07-24": "凱米", "2024-07-25": "凱米",
           "2024-10-02": "山陀兒", "2024-10-03": "山陀兒", "2024-10-31": "康芮"}


def build_calendar() -> pd.DataFrame:
    cal = pd.DataFrame({"date": pd.date_range("2023-01-01", "2026-05-31")})
    s = cal.date.dt.strftime("%Y-%m-%d")
    cal["dow"] = cal.date.dt.dayofweek
    cal["month"] = cal.date.dt.month
    cal["is_weekend"] = (cal.dow >= 5).astype(int)
    cal["is_holiday"] = s.isin(WEEKDAY_HOLIDAYS).astype(int)
    cal["is_makeup_workday"] = s.isin(MAKEUP_WORKDAYS).astype(int)
    cal["is_offday"] = (((cal.is_weekend == 1) & (cal.is_makeup_workday == 0))
                        | (cal.is_holiday == 1)).astype(int)
    cal["holiday_name"] = s.map(WEEKDAY_HOLIDAYS).fillna(s.map(WEEKEND_NAMES)).fillna("")
    lny = set()
    for a, b in LNY_SPANS:
        lny.update(d.strftime("%Y-%m-%d") for d in pd.date_range(a, b))
    cal["is_lny"] = s.isin(lny).astype(int)
    cal["is_typhoon"] = s.isin(TYPHOON).astype(int)
    cal["typhoon_name"] = s.map(TYPHOON).fillna("")
    cal["is_nye"] = ((cal.month == 12) & (cal.date.dt.day == 31)).astype(int)
    cal.to_csv(FEAT / "calendar_daily.csv", index=False)
    return cal


def kmeans(X, K, seeds=8, iters=200):
    best = None
    for seed in range(seeds):
        r = np.random.default_rng(seed)
        C = X[r.choice(len(X), K, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
            if np.allclose(C2, C):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def build_static() -> pd.DataFrame:
    """站點靜態特徵：僅用訓練期（~TRAIN_END）避免洩漏。"""
    con = duckdb.connect()
    fix_sql = "CASE destination " + " ".join(
        f"WHEN '{k}' THEN '{v}'" for k, v in FIX.items()) + " ELSE destination END"
    od = con.execute(f"""
        SELECT origin, {fix_sql} AS destination, SUM(trips) AS t
        FROM read_parquet('{PQ}') WHERE date <= '{TRAIN_END}' GROUP BY 1, 2
    """).fetchdf()
    # 分時剖面排除延長營運日（凌晨02-04時有流量者）。全期共6天，均可對應到已知活動：
    #   2024-01-01 / 2025-01-01 / 2026-01-01 → 跨年
    #   2023-10-07 / 2024-11-02 / 2025-11-01 → 臺北白晝之夜（Nuit Blanche，通宵藝術節）
    # 採規則式偵測而非寫死日期，資料更新後新增的延長營運日會自動被涵蓋。
    # 正常營運不跨越該時段，故不排除時 COUNT(DISTINCT date) 會塌縮成個位數，
    # 把每年三天的跨年人潮放大成假的日常訊號，並捏造出一個偽群。
    # 事件資料本身完整保留於日層級特徵（is_nye）與 Stage 8 事件分析。
    prof = con.execute(f"""
        WITH ext AS (SELECT DISTINCT date FROM read_parquet('{PQ}')
                     WHERE hour BETWEEN 2 AND 4 AND trips > 0)
        SELECT origin AS station, hour,
               CASE WHEN dayofweek(date) IN (0,6) THEN 1 ELSE 0 END AS wend,
               SUM(trips)::DOUBLE / COUNT(DISTINCT date) AS avg_trips
        FROM read_parquet('{PQ}')
        WHERE date <= '{TRAIN_END}' AND date NOT IN (SELECT date FROM ext)
        GROUP BY 1, 2, 3
    """).fetchdf()
    con.close()

    sts = sorted(set(od.origin))
    idx = {s: i for i, s in enumerate(sts)}
    n = len(sts)
    A = np.zeros((n, n))
    dmask = od.destination.isin(idx)
    A[od.origin[dmask].map(idx), od.destination[dmask].map(idx)] = od.t[dmask]
    outs, ins = A.sum(1), A.sum(0)
    P = A / np.where(outs[:, None] == 0, 1, outs[:, None])
    pr = np.ones(n) / n
    for _ in range(200):
        pr = .15 / n + .85 * (P.T @ pr)

    pv = prof.pivot_table(index="station", columns=["wend", "hour"], values="avg_trips",
                          fill_value=0).reindex(sts).fillna(0)
    X = pv.div(pv.sum(1).replace(0, 1), axis=0).values
    lab, _ = kmeans(X, 4)
    # 以小時標籤取尖峰：排除延長營運日後平日只剩21個時段（02-04時消失），
    # 位置索引 m[7:10] 會取到錯誤的小時。
    hpos = {h: i for i, h in enumerate(pv[0].columns)}
    AM = [hpos[h] for h in (7, 8, 9) if h in hpos]
    PM = [hpos[h] for h in (17, 18, 19) if h in hpos]
    wk = pv[0].values
    wkn = wk / np.where(wk.sum(1, keepdims=True) == 0, 1, wk.sum(1, keepdims=True))
    names = {}
    for k in range(4):
        m = wkn[lab == k].mean(0)
        am, pm = m[AM].sum(), m[PM].sum()
        wshare = pv[1].values[lab == k].sum() / max(pv.values[lab == k].sum(), 1)
        if am > pm * 1.25:
            names[k] = "residential"
        elif pm > am * 1.25:
            # 晚峰群再依尖峰強度細分：17-19時占平日>45%者為極端單峰的純就業腹地
            # （松江南京、港墘、西湖、南港軟體園區等，實測53%vs一般就業群37%）。
            # 未細分時兩群會共用"employment"標籤，使k=4實際只產出3個類別。
            names[k] = "employment_peak" if pm > .45 else "employment"
        elif wshare > .40:
            names[k] = "tourism_hub"
        else:
            names[k] = "mixed"
    assert len(set(names.values())) == 4, f"分群標籤重複：{list(names.values())}"
    static = pd.DataFrame({
        "station": sts, "cluster": [names[k] for k in lab], "pagerank": pr,
        "out_strength_train": outs.astype(int), "in_strength_train": ins.astype(int),
        "deg_out": (A > 0).sum(1), "deg_in": (A > 0).sum(0)})
    static.to_csv(FEAT / "station_static.csv", index=False)
    return static


def build_feature_table(cal: pd.DataFrame, static: pd.DataFrame):
    sd = pd.read_csv(AGG / "station_daily.csv", parse_dates=["date"])[["date", "station", "entries"]]
    sd = sd[sd.station.isin(set(static.station))]
    # 完整網格：缺列=真實零進站（如颱風日/停駛日的小站）
    full = pd.MultiIndex.from_product(
        [sorted(static.station), pd.date_range("2023-01-01", "2026-05-31")],
        names=["station", "date"]).to_frame(index=False)
    df = full.merge(sd, on=["station", "date"], how="left")
    df["entries"] = df.entries.fillna(0).astype(int)

    df = df.merge(cal, on="date", how="left").sort_values(["station", "date"]).reset_index(drop=True)
    g = df.groupby("station")["entries"]
    df["lag1"] = g.shift(1)
    df["lag7"] = g.shift(7)
    df["lag14"] = g.shift(14)
    df["roll7_mean"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll28_mean"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df = df.merge(static, on="station", how="left")
    df["is_yline"] = df.station.isin(YLINE).astype(int)
    # 環狀線0403地震停駛旗標——近似編碼。實際時間軸：2024/4/3 07:58全線停駛；
    # 板橋—新北產業園區段5站當日17:00即恢復（單線雙向、班距16分，其後長期減班營運）；
    # 中和—大坪林段6站4/7 11:00恢復；受損段板新/中原/橋和停駛至12/12中午全線通車。
    # 本旗標以「全線2024/4/4-4/6＋受損三站至2024/12/11」近似：板橋以西5站於4/4-4/6
    # 被誤標為停駛（15站日，佔全表約0.01%），全落在訓練期、test期恆為0，不影響任何結果。
    # 來源：新北捷運工程局 https://www.dorts.ntpc.gov.tw/news/indexInfo/MJjdJpJYmAv6?page=2
    #       公視新聞 https://news.pts.org.tw/article/688630
    #       中央社 https://www.cna.com.tw/news/ahel/202404070033.aspx
    #       中央社 https://www.cna.com.tw/news/ahel/202412120040.aspx
    df["service_suspended"] = 0
    m = (df.station.isin(YLINE) & df.date.between("2024-04-04", "2024-04-06")) | \
        (df.station.isin(["中原", "板新", "橋和"]) & df.date.between("2024-04-04", "2024-12-11"))
    df.loc[m, "service_suspended"] = 1
    df.to_csv(FEAT / "feature_table.csv", index=False)
    return df


def validate(df: pd.DataFrame):
    assert df.entries.isna().sum() == 0
    assert df.groupby("station").size().nunique() == 1
    assert df[df.is_typhoon == 1].date.nunique() == 6
    assert df[df.is_makeup_workday == 1].date.nunique() == 8
    print("驗證通過：", df.shape,
          f"| 颱風日/平日運量比 {df[df.is_typhoon==1].entries.mean()/df[(df.is_typhoon==0)&(df.is_offday==0)].entries.mean():.1%}")


if __name__ == "__main__":
    cal = build_calendar()
    static = build_static()
    df = build_feature_table(cal, static)
    validate(df)
    print("完成：", FEAT)
