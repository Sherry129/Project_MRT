# -*- coding: utf-8 -*-
"""
階段2：探索性資料分析（EDA）
================================
輸入：data/parquet/od/od_YYYYMM.parquet（stage1b產出）
輸出：data/agg/*.csv 聚合表 + reports/figures/fig1~fig5.png（含fig3b k值診斷）

分時剖面的資料處理原則：
「延長營運日」僅排除於 station_hour_daytype.csv，因為分群要描述的是典型日常節奏，
不是事件行為。其餘聚合表完整保留這些日子，事件本身由 is_nye 特徵與 Stage 8 專責分析。

偵測規則：凌晨02-04時有流量者。正常營運不跨越該時段，故此條件等價於「當日延長營運」。
全期命中6天（佔1247天的0.48%），均可對應到已知活動：
  2024-01-01 / 2025-01-01 / 2026-01-01 → 跨年
  2023-10-07 / 2024-11-02 / 2025-11-01 → 臺北白晝之夜（Nuit Blanche，通宵藝術節）
採規則式偵測而非寫死日期，資料更新後新增的延長營運日會自動被涵蓋。
排除清單見 data/agg/extended_service_days.csv。

分析內容：
1. 系統日運量趨勢與極端事件日
2. 平日/週末分時曲線
3. 站點功能分群（k-means on 進站曲線形狀，k值以elbow/silhouette/gap三指標驗證）
4. OD矩陣熱圖與前15大OD對
5. 網絡中心性（PageRank）

安裝需求：pip install duckdb pandas matplotlib scikit-learn
用法：python scripts/stage2_eda.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
PQ = ROOT / "data" / "parquet" / "od"
AGG = ROOT / "data" / "agg"
FIG = ROOT / "reports" / "figures"
AGG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# 202305過渡月的站名前綴變體 → 正規化
FIX = {"G大坪林": "大坪林", "O景安": "景安", "O頭前庄": "頭前庄"}

# 圖1極端值標註用：日運量最高/最低前三名的事件對照（人工核對自 reports/stage2_eda_report.md 第2節，
# 與 stage4_features.py 的 TYPHOON/is_nye 定義一致）。僅供圖1標籤，非完整日曆特徵。
EVENTS = {
    "2024-12-31": "跨年夜", "2025-12-31": "跨年夜", "2025-11-14": "耶誕城開幕",
    "2024-10-31": "康芮颱風", "2024-07-24": "凱米颱風", "2024-10-03": "山陀兒颱風",
}


def _fig1_label(d):
    """日期加上事件名（查不到就只印日期）。"""
    ev = EVENTS.get(d.strftime("%Y-%m-%d"))
    return f"{d:%Y-%m-%d} {ev}" if ev else f"{d:%Y-%m-%d}"


def _annotate_side(ax, d, y, x0, span_days, color):
    """把標註放在資料點的左側或右側（垂直置中於資料點本身的y值），
    而非上/下方，這樣文字的垂直位置必落在座標軸範圍內，不需額外留白。
    日期落在前半段就掛右邊、後半段掛左邊，避免文字被畫布邊緣切掉。"""
    frac = (d - x0).days / span_days
    right = frac < 0.5
    ax.annotate(_fig1_label(d), (d, y), fontsize=7, va="center",
                ha=("left" if right else "right"),
                xytext=(8 if right else -8, 0), textcoords="offset points",
                color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=.7, pad=1))


def _repel_labels(ax, xs, ys, labels, fontsize=7.5, dx=13, dy=7, line_gap=1.25):
    """散點標註防重疊：標籤一律掛在資料點的左上方（此圖散點沿對角線單調遞增，
    左上必為空白區），再於display座標由下而上把重疊的標籤逐一往上推開，最後以
    細引線連回資料點——位移後仍看得出標籤屬於哪一點。"""
    fig = ax.figure
    fig.canvas.draw()  # 先定稿座標轉換，否則transData尚未反映最終軸範圍
    P = ax.transData.transform(np.column_stack([xs, ys]))
    row_h = fontsize * fig.dpi / 72 * line_gap  # 單行字高（含行距）
    order = np.argsort(P[:, 1])  # 由下而上處理
    ty = P[order, 1] + dy
    for i in range(1, len(ty)):
        ty[i] = max(ty[i], ty[i - 1] + row_h)  # 與下方鄰居至少隔一行
    inv = ax.transData.inverted()
    for k, i in enumerate(order):
        # xytext回存成data座標，後續tight_layout重排版面時標籤才不會脫離資料點
        ax.annotate(labels[i], (xs[i], ys[i]), textcoords="data",
                    xytext=inv.transform((P[i, 0] - dx, ty[k])),
                    fontsize=fontsize, ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=.5, color="#888", shrinkA=0, shrinkB=2))


def build_aggregates():
    """用DuckDB一次掃描所有月檔產出聚合表（不載入原始資料進記憶體）。"""
    con = duckdb.connect()
    src = f"read_parquet('{(PQ / 'od_*.parquet').as_posix()}')"
    fix_sql = "CASE destination " + " ".join(
        f"WHEN '{k}' THEN '{v}'" for k, v in FIX.items()) + " ELSE destination END"

    con.execute(f"""
        COPY (
            WITH base AS (SELECT date, hour, origin, {fix_sql} AS destination, trips FROM {src})
            SELECT date, station, SUM(entries) AS entries, SUM(exits) AS exits FROM (
                SELECT date, origin AS station, trips AS entries, 0 AS exits FROM base
                UNION ALL
                SELECT date, destination, 0, trips FROM base
            ) GROUP BY date, station ORDER BY date, station
        ) TO '{AGG / 'station_daily.csv'}' (HEADER)
    """)
    # 延長營運日偵測：正常營運不跨越02-04時，故該時段有流量者必為跨年/大型活動的延長營運。
    # 這些日子只排除於「分時剖面」聚合，其餘聚合表（日運量、OD矩陣、系統時流量）完整保留。
    ext = f"(SELECT DISTINCT date FROM {src} WHERE hour BETWEEN 2 AND 4 AND trips > 0)"
    con.execute(f"""
        COPY (SELECT date, SUM(trips) AS trips_0204 FROM {src}
              WHERE hour BETWEEN 2 AND 4 AND date IN {ext}
              GROUP BY 1 ORDER BY 1)
        TO '{AGG / 'extended_service_days.csv'}' (HEADER)
    """)
    # 分時剖面：排除延長營運日。若不排除，02-04時僅那幾天有資料列，
    # COUNT(DISTINCT date) 會塌縮成個位數，把跨年人潮放大成假的日常訊號
    # （象山02時 22.7 → 6739 人次，放大297倍），並在k=4時捏造出一個
    # 由信義線東段站點組成的偽群。排除後各cell覆蓋率均為1.0。
    con.execute(f"""
        COPY (
            SELECT origin AS station, hour,
                   CASE WHEN dayofweek(date) IN (0,6) THEN 'weekend' ELSE 'weekday' END AS daytype,
                   SUM(trips) AS total_trips,
                   ROUND(SUM(trips)::DOUBLE / COUNT(DISTINCT date), 2) AS avg_per_day
            FROM {src} WHERE date NOT IN {ext}
            GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        ) TO '{AGG / 'station_hour_daytype.csv'}' (HEADER)
    """)
    con.execute(f"""
        COPY (
            SELECT origin, {fix_sql} AS destination, SUM(trips) AS total_trips
            FROM {src} GROUP BY 1, 2 HAVING SUM(trips) > 0 ORDER BY 3 DESC
        ) TO '{AGG / 'od_matrix_total.csv'}' (HEADER)
    """)
    con.execute(f"""
        COPY (
            SELECT date, hour, SUM(trips) AS trips FROM {src} GROUP BY 1, 2 ORDER BY 1, 2
        ) TO '{AGG / 'system_hourly.csv'}' (HEADER)
    """)
    con.execute(f"""
        COPY (
            SELECT strftime(date, '%Y%m') AS month,
                   COUNT(DISTINCT origin) AS n_origin, COUNT(DISTINCT destination) AS n_dest
            FROM {src} GROUP BY 1 ORDER BY 1
        ) TO '{AGG / 'month_station_coverage.csv'}' (HEADER)
    """)
    con.close()
    print("聚合表已輸出至", AGG)


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
    return best[1], best[2], best[0]


def silhouette(X, lab):
    """平均輪廓係數：(b-a)/max(a,b)，a=群內平均距離、b=最近他群平均距離。範圍[-1,1]，越高越好。"""
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        if own.sum() <= 1:
            continue
        a = D[i][own].sum() / (own.sum() - 1)
        b = min(D[i][lab == k].mean() for k in set(lab) if k != lab[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


def cluster_diagnostics(X, ks=range(2, 11), B=20, seed=0):
    """k值選擇診斷：對每個k同時計算三個互補指標，避免只靠肉眼看手肘。

    inertia   ：群內平方和（elbow method用），必然隨k單調下降，看的是「降幅何時趨緩」
    silhouette：輪廓係數，衡量群的分離度，可直接取最大值
    gap       ：Gap統計量，與均勻分布虛無參考比較；準則為取最小的k使 gap(k) >= gap(k+1)-s(k+1)
    """
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    rows = []
    for K in ks:
        lab, _, inertia = kmeans(X, K)
        ref = [np.log(kmeans(rng.uniform(lo, hi, X.shape), K, seeds=2)[2]) for _ in range(B)]
        rows.append(dict(k=K, inertia=inertia, silhouette=silhouette(X, lab),
                         gap=np.mean(ref) - np.log(inertia),
                         gap_sd=np.std(ref) * np.sqrt(1 + 1 / B)))
    d = pd.DataFrame(rows)
    # elbow量化：二階差分越大代表曲線在該點轉折越劇烈（端點無法計算，故留NaN）
    d["curvature"] = np.nan
    d.loc[d.index[1:-1], "curvature"] = (d.inertia.values[:-2] - 2 * d.inertia.values[1:-1]
                                         + d.inertia.values[2:])
    d["drop_pct"] = -d.inertia.pct_change() * 100
    return d


def plot_diagnostics(d, k_used, path):
    """把三個指標畫成一張圖，作為k值選擇的佐證。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    k_elbow = int(d.loc[d.curvature.idxmax(), "k"])
    axes[0].plot(d.k, d.inertia, "-o", color="#4878CF")
    axes[0].axvline(k_elbow, ls="--", color="#D65F5F", label=f"最大曲率 k={k_elbow}")
    axes[0].axvline(k_used, ls=":", color="k", label=f"實際採用 k={k_used}")
    axes[0].set_ylabel("群內平方和 (inertia)"); axes[0].set_title("Elbow Method")
    axes[1].plot(d.k, d.silhouette, "-o", color="#6ACC65")
    axes[1].axvline(int(d.loc[d.silhouette.idxmax(), "k"]), ls="--", color="#D65F5F",
                    label=f"最大值 k={int(d.loc[d.silhouette.idxmax(), 'k'])}")
    axes[1].axvline(k_used, ls=":", color="k", label=f"實際採用 k={k_used}")
    axes[1].set_ylabel("平均輪廓係數"); axes[1].set_title("Silhouette Score")
    axes[2].errorbar(d.k, d.gap, yerr=d.gap_sd, fmt="-o", color="#B47CC7", capsize=3)
    axes[2].axvline(k_used, ls=":", color="k", label=f"實際採用 k={k_used}")
    axes[2].set_ylabel("Gap 統計量"); axes[2].set_title("Gap Statistic（單調上升＝無明確分群數）")
    for ax in axes:
        ax.set_xlabel("群數 k"); ax.set_xticks(d.k); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("站點分群 k 值選擇診斷", y=1.03)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return k_elbow


def make_figures():
    # 圖1 系統日運量
    sh = pd.read_csv(AGG / "system_hourly.csv", parse_dates=["date"])
    daily = sh.groupby("date")["trips"].sum()
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(daily.index, daily / 1e6, lw=.5, alpha=.55, color="#4878CF", label="單日")
    ax.plot(daily.index, daily.rolling(7, center=True).mean() / 1e6, lw=1.8, color="#D65F5F", label="7日移動平均")
    x0, span_days = daily.index.min(), (daily.index.max() - daily.index.min()).days
    for d, v in daily.nlargest(3).items():
        _annotate_side(ax, d, v / 1e6, x0, span_days, "darkred")
    for d, v in daily.nsmallest(3).items():
        _annotate_side(ax, d, v / 1e6, x0, span_days, "darkblue")
    ax.set_ylabel("日總運量（百萬人次）"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("台北捷運系統日運量 2023-01 ~ 2026-05")
    fig.tight_layout(); fig.savefig(FIG / "fig1_daily_ridership.png", dpi=130); plt.close(fig)

    # 圖1（無標註版）：同一份資料，不畫日期/事件標註
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(daily.index, daily / 1e6, lw=.5, alpha=.55, color="#4878CF", label="單日")
    ax.plot(daily.index, daily.rolling(7, center=True).mean() / 1e6, lw=1.8, color="#D65F5F", label="7日移動平均")
    ax.set_ylabel("日總運量（百萬人次）"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("台北捷運系統日運量 2023-01 ~ 2026-05")
    fig.tight_layout(); fig.savefig(FIG / "fig1_nomark.png", dpi=130); plt.close(fig)

    # 圖2 平假日分時曲線
    hd = pd.read_csv(AGG / "station_hour_daytype.csv")
    sys_h = hd.groupby(["daytype", "hour"])["avg_per_day"].sum().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(sys_h.index, sys_h["weekday"] / 1e3, "-o", ms=3.5, color="#4878CF", label="平日")
    ax.plot(sys_h.index, sys_h["weekend"] / 1e3, "-s", ms=3.5, color="#D65F5F", label="週末")
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("時段"); ax.set_ylabel("平均進站量（千人次/小時）")
    ax.set_title("系統平均分時進站量：平日 vs 週末"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig2_hourly_profile.png", dpi=130); plt.close(fig)

    # 圖3 站點分群
    pv = hd.pivot_table(index="station", columns=["daytype", "hour"], values="avg_per_day", fill_value=0)
    pv = pv[sorted(pv.columns)]
    # profile_scale：42維剖面向量的L1範數（＝平日日均進站＋週末日均進站），
    # 只用來把每站曲線normalize成「各時段佔比」，本身不是任何有意義的運量指標。
    # 注意：它約為真實日均的兩倍，不可拿來當「該站日均進站量」引用。
    profile_scale = pv.sum(axis=1)
    X = pv.div(profile_scale, axis=0).values
    # daily_avg：該站真實日均進站量（全期1,247天的entries算術平均），
    # 供station_clusters.csv輸出與圖3標題挑代表站使用。
    daily_avg = (pd.read_csv(AGG / "station_daily.csv")
                 .groupby("station")["entries"].mean().reindex(pv.index))
    # 圖3b k值選擇診斷（elbow / silhouette / gap）
    diag = cluster_diagnostics(X)
    diag.round(5).to_csv(AGG / "cluster_k_diagnostics.csv", index=False)
    K = 4
    k_elbow = plot_diagnostics(diag, K, FIG / "fig3b_k_selection.png")
    print("\n=== k值選擇診斷 ===")
    print(diag.round(4).to_string(index=False))
    print(f"elbow(最大曲率)建議 k={k_elbow}；"
          f"silhouette最大在 k={int(diag.loc[diag.silhouette.idxmax(), 'k'])}；"
          f"gap統計量單調上升未觸發停止準則 → 資料為連續光譜、無統計上唯一的最佳k。")
    print(f"採用 k={K}：在elbow平緩區內，且為能分離出「極端單一晚峰」型態的最小k。"
          f"消融實驗證實k=3/k=4對預測效能無顯著差異（見 reports/stage6b_k_ablation.md）。\n")

    lab, C, _ = kmeans(X, K)
    # 以「小時標籤」而非位置索引取尖峰：排除延長營運日後平日只剩21個時段
    # （02-04時消失），用 m[7:10] 這種位置切片會取到錯誤的小時。
    wk_hours = list(pv["weekday"].columns)
    hpos = {h: i for i, h in enumerate(wk_hours)}
    AM = [hpos[h] for h in (7, 8, 9) if h in hpos]
    PM = [hpos[h] for h in (17, 18, 19) if h in hpos]
    wk = pv["weekday"].values
    wk_norm = wk / wk.sum(1, keepdims=True)
    names = {}
    for k in range(K):
        m = wk_norm[lab == k].mean(0)
        am, pm = m[AM].sum(), m[PM].sum()
        wend_share = pv["weekend"].values[lab == k].sum() / pv.values[lab == k].sum()
        if am > pm * 1.25:
            names[k] = "住宅型（早進站尖峰）"
        elif pm > am * 1.25:
            # 晚峰群再依尖峰強度細分：>45%者為極端單峰的純就業腹地，避免兩群共用同一標籤
            # （成員為松江南京、港墘、西湖、南港軟體園區、先嗇宮、中原、橋和，
            #   共同點是曲線形狀而非地理位置，故標籤不作地理宣稱）
            names[k] = ("就業極端型（單一晚峰）" if pm > .45
                        else "就業型（晚進站尖峰）")
        elif wend_share > .40:
            names[k] = "轉運/觀光休閒型（週末權重高）"
        else:
            names[k] = "混合型"
    assert len(set(names.values())) == K, f"k={K} 產生重複群標籤：{list(names.values())}"
    fig, axes = plt.subplots(1, K, figsize=(16, 3.6), sharey=True)
    colors = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7"]
    nwk = len(wk_hours)   # 平日欄數，不可寫死24
    for k, ax in enumerate(axes):
        mem = np.where(lab == k)[0]
        for i in mem:
            ax.plot(wk_hours, X[i][:nwk] * 100, color=colors[k], alpha=.25, lw=.7)
        ax.plot(wk_hours, C[k][:nwk] * 100, color="k", lw=2)
        big = daily_avg.iloc[mem].nlargest(3).index.tolist()
        ax.set_title(f"{names[k]}\nn={len(mem)}｜例：{'、'.join(big)}", fontsize=9)
        ax.set_xlabel("時段（平日）"); ax.grid(alpha=.3)
    axes[0].set_ylabel("占全日比例 (%)")
    fig.suptitle("站點分群：平日分時進站曲線形狀（k-means, k=4）", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig3_station_clusters.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame({"station": pv.index, "cluster": [names[k] for k in lab],
                  "daily_avg": daily_avg.round(0)}).sort_values(
        ["cluster", "daily_avg"], ascending=[True, False]).to_csv(AGG / "station_clusters.csv", index=False)

    # 圖3c 同一組分群結果，改畫假日（週末）分時曲線；X/C的後半段本來就是週末部分，不必重新分群
    we_hours = list(pv["weekend"].columns)
    nwe = len(we_hours)
    fig, axes = plt.subplots(1, K, figsize=(16, 3.6), sharey=True)
    for k, ax in enumerate(axes):
        mem = np.where(lab == k)[0]
        for i in mem:
            ax.plot(we_hours, X[i][nwk:nwk + nwe] * 100, color=colors[k], alpha=.25, lw=.7)
        ax.plot(we_hours, C[k][nwk:nwk + nwe] * 100, color="k", lw=2)
        big = daily_avg.iloc[mem].nlargest(3).index.tolist()
        ax.set_title(f"{names[k]}\nn={len(mem)}｜例：{'、'.join(big)}", fontsize=9)
        ax.set_xlabel("時段（假日）"); ax.grid(alpha=.3)
    axes[0].set_ylabel("占全日比例 (%)")
    fig.suptitle("站點分群：假日分時進站曲線形狀（沿用平日k-means分群結果，k=4）", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "fig3c_station_clusters_weekend.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # 圖4 OD熱圖
    # 先合併站名前綴變體再繪圖：2023-05 環狀線移交過渡月產生 G大坪林／O景安／O頭前庄
    # 三個標籤，它們只出現在出站側、從不作為進站側，因此無法進入「前40大進站站」名單
    # 而被排除在熱圖之外。其中景安憑進站側流量入選 top40，若不合併，其橫列（進站）
    # 資料完整、直行（出站）卻只剩實際值約12%，形成「很多人出發、幾乎沒人在此出站」
    # 的假象。合併後三站的出站側序列連續（前綴前後每月量級一致），資料方為完整。
    # 對照表與 stage4_features.py 的 FIX 相同。
    od = pd.read_csv(AGG / "od_matrix_total.csv")
    od["destination"] = od.destination.replace({"G大坪林": "大坪林", "O景安": "景安",
                                                "O頭前庄": "頭前庄"})
    od = od.groupby(["origin", "destination"], as_index=False)["total_trips"].sum()
    top40 = od.groupby("origin")["total_trips"].sum().nlargest(40).index
    M = od[od.origin.isin(top40) & od.destination.isin(top40)].pivot_table(
        index="origin", columns="destination", values="total_trips", fill_value=0)
    order = M.sum(1).sort_values(ascending=False).index
    M = M.loc[order, [c for c in order if c in M.columns]]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(np.log10(M.values + 1), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index, fontsize=6.5)
    fig.colorbar(im, label="log10(累計人次+1)", shrink=.7)
    ax.set_title("OD累計流量熱圖（進出站量前40大站）"); ax.set_xlabel("出站"); ax.set_ylabel("進站")
    fig.tight_layout(); fig.savefig(FIG / "fig4_od_heatmap.png", dpi=130); plt.close(fig)

    # 圖5 top OD對 + PageRank
    od2 = od[od.origin != od.destination]
    top15 = od2.nlargest(15, "total_trips")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    lbl = top15.origin + " → " + top15.destination
    ax1.barh(lbl[::-1], top15.total_trips[::-1] / 1e6, color="#4878CF")
    ax1.set_xlabel("累計人次（百萬）"); ax1.set_title("前15大OD流量對")
    sts = sorted(set(od.origin) | set(od.destination))
    idx = {s: i for i, s in enumerate(sts)}
    A = np.zeros((len(sts), len(sts)))
    A[od.origin.map(idx), od.destination.map(idx)] = od.total_trips
    outs = A.sum(1)
    P = A / np.where(outs[:, None] == 0, 1, outs[:, None])
    pr = np.ones(len(sts)) / len(sts)
    for _ in range(100):
        pr = .15 / len(sts) + .85 * (P.T @ pr)
    ax2.scatter(outs / 1e6, pr * 100, s=14, alpha=.6, color="#6ACC65")
    top = np.argsort(pr)[-8:]
    _repel_labels(ax2, outs[top] / 1e6, pr[top] * 100, [sts[i] for i in top])
    ax2.set_xlabel("累計進站量（百萬）"); ax2.set_ylabel("PageRank (%)")
    ax2.set_title("站點網絡中心性：PageRank vs 進站量")
    for a in (ax1, ax2):
        a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig5_od_pairs_centrality.png", dpi=130); plt.close(fig)
    print("圖表已輸出至", FIG)


if __name__ == "__main__":
    build_aggregates()
    make_figures()
