# -*- coding: utf-8 -*-
"""
階段8：SHAP可解釋性 + 事件case study
================================
輸入：data/features/feature_table.csv（stage4）
     reports/stage6_tuning_log.csv（LightGBM best params，自動讀取）
     reports/stage5_sarima_predictions.csv / stage6_ml_predictions.csv / stage7_dl_predictions.csv
輸出：reports/stage8_shap_importance.csv      （全域+按站型 mean|SHAP|）
     reports/stage8_event_group_errors.csv   （test期各事件日型×模型 WAPE）
     reports/stage8_worst_stations.csv       （SARIMAX最差5站的三世代對照）
     reports/stage8_holdout_events.csv       （颱風/耶誕城/跨年 hold-out實驗）
     reports/figures/fig6~fig10.png

分析設計（依handoff §6 Stage 8規劃）：
A. SHAP作用在Stage 6最佳樹模型（LightGBM, train+val重訓，與stage6完全同規格）
   A1 全域重要度（mean|SHAP|，在test集上計算）
   A2 按站型分組的mean|SHAP|佔比 → 回答「不同站型依賴什麼特徵」（研究問題c的機制面）
   A3 事件日local explanation（除夕、元旦等，個站×個日的特徵貢獻拆解）
B. test期（2026/01-05）事件日誤差：三世代模型（SARIMAX/LightGBM/LSTM）
   按日型（平日/週末/國定假日/春節/元旦/補班）分組WAPE + 每模型最差5日
   + SARIMAX最差5站（事件型場站）的三世代對照
C. Hold-out事件實驗（研究問題b：事件特徵的挽回效果）
   C1 颱風4事件6天：訓練截至事件前一日，比較 有/無is_typhoon 的LightGBM
      （卡努=zero-shot、凱米=看過1次、山陀兒=2次、康芮=3次 → 旗標效果隨經驗變化）
   C2 新北耶誕城開幕2025-11-14：無任何特徵編碼此事件 → 量化BL板橋低估幅度（資訊缺口示範）
   C3 跨年2025-12-31：有/無is_nye 的系統級WAPE差（旗標已看過2次跨年）
   C4 0403地震：不做retrain——樹模型對「訓練期從未見過=1」的旗標無法外推
      （service_suspended在2024-04-04前全為0，切分點不存在），此限制直接寫進報告。

防洩漏備忘：所有hold-out實驗訓練資料嚴格 < 事件首日；事件旗標本身是事前可得資訊。

安裝需求：pip install shap（其餘同stage6：pandas numpy scikit-learn lightgbm matplotlib）
用法：python scripts/stage8_shap_events.py [--skip-shap] [--skip-holdout]
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC",
                                   "Noto Sans CJK JP", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "feature_table.csv"
REPORTS = ROOT / "reports"
FIGDIR = REPORTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2024-12-31"
VAL_END = "2025-12-31"

# 與stage6_ml.py完全一致的特徵集
CAL_FEATS = ["dow", "month", "is_weekend", "is_holiday", "is_offday",
             "is_makeup_workday", "is_lny", "is_typhoon", "is_nye"]
LAG_FEATS = ["lag1", "lag7", "lag14", "roll7_mean", "roll28_mean"]
NET_FEATS = ["pagerank", "out_strength_train", "in_strength_train", "deg_out", "deg_in"]
OTHER_STATIC = ["is_yline", "service_suspended"]

CN = {"residential": "住宅", "employment": "就業", "employment_peak": "就業極端",
      "tourism_hub": "觀光", "mixed": "混合"}


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return np.abs(y - p).sum() / y.sum() * 100


def load_data():
    df = pd.read_csv(FEAT, parse_dates=["date"])
    df = df.dropna(subset=LAG_FEATS).reset_index(drop=True)
    df = pd.concat([df, pd.get_dummies(df.cluster, prefix="cl").astype(int)], axis=1)
    cl_cols = [c for c in df.columns if c.startswith("cl_")]
    return df, cl_cols


def best_params():
    """讀stage6調參紀錄取LightGBM full最佳列；讀不到就用報告記載的值。"""
    try:
        log = pd.read_csv(REPORTS / "stage6_tuning_log.csv")
        row = (log[(log.model == "LightGBM") & (log.ablation == "full")]
               .sort_values("val_WAPE").iloc[0])
        return dict(num_leaves=int(row.num_leaves), min_child_samples=int(row.min_child_samples),
                    learning_rate=float(row.learning_rate), n_estimators=int(row.best_n_estimators))
    except Exception:
        print("  [warn] 讀不到stage6_tuning_log.csv，改用報告記載的best params")
        return dict(num_leaves=255, min_child_samples=50, learning_rate=0.05, n_estimators=211)


def fit_lgbm(X, y, params):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **params)
    m.fit(X, y)
    return m


def find_station(df, name):
    """站名容錯：先精確再子字串（板橋等有線別前綴）。"""
    st = df.station.unique()
    if name in st:
        return name
    hit = [s for s in st if name in s]
    return hit[0] if hit else None


# ================================================================ A. SHAP
def part_a_shap(df, cl_cols, params):
    import shap
    print("\n========== A. SHAP（LightGBM, train+val重訓, 於test集計算）==========")
    feats = CAL_FEATS + LAG_FEATS + NET_FEATS + OTHER_STATIC + cl_cols
    trva = df[df.date <= VAL_END]
    te = df[df.date > VAL_END].reset_index(drop=True)
    m = fit_lgbm(trva[feats].values.astype(float), trva.entries.values.astype(float), params)
    p = np.maximum(m.predict(te[feats].values.astype(float)), 0)
    print(f"  sanity check：test WAPE = {wape(te.entries, p):.3f}%（stage6報告=4.62%，應一致）")

    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(te[feats].values.astype(float))
    sv = np.asarray(sv)
    base = float(np.ravel(expl.expected_value)[0])

    # ---- A1 全域重要度
    rows = [{"feature": f, "group": "ALL", "mean_abs_shap": a}
            for f, a in zip(feats, np.abs(sv).mean(0))]
    # ---- A2 按站型
    for cl, sub in te.groupby("cluster"):
        idx = sub.index.values
        for f, a in zip(feats, np.abs(sv[idx]).mean(0)):
            rows.append({"feature": f, "group": cl, "mean_abs_shap": a})
    imp = pd.DataFrame(rows)
    # 每群內轉成佔比%（不同站型量級不同，佔比才可比）
    imp["share_pct"] = imp.groupby("group").mean_abs_shap.transform(lambda s: s / s.sum() * 100)
    imp.to_csv(REPORTS / "stage8_shap_importance.csv", index=False)

    g = imp[imp.group == "ALL"].sort_values("mean_abs_shap", ascending=False)
    print("\n  全域 mean|SHAP| Top12：")
    print(g.head(12)[["feature", "mean_abs_shap", "share_pct"]].round(1).to_string(index=False))

    top = g.feature.head(12).tolist()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(range(len(top))[::-1], g.mean_abs_shap.head(12), color="#4878a8")
    ax.set_yticks(range(len(top))[::-1], top)
    ax.set_xlabel("mean |SHAP|（人次）")
    ax.set_title("fig6　LightGBM全域特徵重要度（SHAP, test集）")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_shap_global.png", dpi=150)
    plt.close(fig)

    # fig7：Top10特徵在三站型的佔比對照
    top10 = g.feature.head(10).tolist()
    piv = imp[(imp.group != "ALL") & (imp.feature.isin(top10))] \
        .pivot(index="feature", columns="group", values="share_pct").loc[top10]
    print("\n  按站型 SHAP佔比%（Top10特徵）：")
    print(piv.round(1).to_string())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(top10))
    w = 0.8 / len(piv.columns)
    for i, cl in enumerate(piv.columns):
        ax.bar(x + i * w, piv[cl], w, label=CN.get(cl, cl))
    ax.set_xticks(x + w, top10, rotation=30, ha="right")
    ax.set_ylabel("mean|SHAP| 佔比（%）")
    ax.set_title("fig7　不同站型依賴的特徵（SHAP佔比）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_shap_by_cluster.png", dpi=150)
    plt.close(fig)

    # ---- A3 事件日 local explanation（除夕、元旦）
    lny_days = te[(te.is_lny == 1)].date
    eve = lny_days.min()  # 2026除夕
    cases = []
    for st_name, day, label in [("台北車站", eve, "除夕"),
                                ("動物園", eve, "除夕"),
                                ("BL板橋", pd.Timestamp("2026-01-01"), "元旦（跨年翌日）")]:
        st = find_station(te, st_name)
        r = te[(te.station == st) & (te.date == day)]
        if len(r) == 0:
            continue
        i = r.index[0]
        cases.append((st, day, label, i))

    fig, axes = plt.subplots(1, len(cases), figsize=(5.5 * len(cases), 5))
    axes = np.atleast_1d(axes)
    print("\n  事件日 local explanation：")
    for ax, (st, day, label, i) in zip(axes, cases):
        contrib = pd.Series(sv[i], index=feats).sort_values(key=np.abs, ascending=False)
        top8 = contrib.head(8)[::-1]
        y_true, y_hat = te.entries.iloc[i], base + sv[i].sum()
        print(f"    {st} {day.date()}（{label}）：實際{y_true:,.0f}，預測{y_hat:,.0f}"
              f"（base {base:,.0f}）；前3貢獻："
              + "、".join(f"{f} {v:+,.0f}" for f, v in contrib.head(3).items()))
        ax.barh(range(len(top8)), top8.values,
                color=["#c0504d" if v > 0 else "#4878a8" for v in top8.values])
        ax.set_yticks(range(len(top8)), top8.index, fontsize=9)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{st}｜{day.date()}（{label}）\n實際{y_true:,.0f}／預測{y_hat:,.0f}", fontsize=10)
        ax.set_xlabel("SHAP貢獻（人次）")
    fig.suptitle("fig8　事件日的特徵貢獻拆解（local SHAP）")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_shap_local_events.png", dpi=150)
    plt.close(fig)


# ================================================================ B. test期事件日誤差
def part_b_events(df):
    print("\n========== B. test期（2026/01-05）事件日誤差：三世代對照 ==========")
    preds = {}
    for tag, fn in [("SARIMAX", "stage5_sarima_predictions.csv"),
                    ("LightGBM", "stage6_ml_predictions.csv"),
                    ("LSTM", "stage7_dl_predictions.csv")]:
        preds[tag] = pd.read_csv(REPORTS / fn, parse_dates=["date"])

    day = (df[df.date > VAL_END]
           .groupby("date")[["is_offday", "is_holiday", "is_lny", "is_weekend",
                             "is_makeup_workday"]].max().reset_index())

    def day_group(r):
        if r.date == pd.Timestamp("2026-01-01"):
            return "元旦(跨年翌日)"
        if r.is_lny:
            return "春節(除夕~初三)"
        if r.is_holiday:
            return "其他國定假日"
        if r.is_makeup_workday:
            return "補班星期六"
        if r.is_weekend:
            return "一般週末"
        return "一般平日"

    day["grp"] = day.apply(day_group, axis=1)
    order = ["一般平日", "一般週末", "補班星期六", "其他國定假日", "春節(除夕~初三)", "元旦(跨年翌日)"]

    rows = []
    for tag, p in preds.items():
        p = p.merge(day[["date", "grp"]], on="date")
        for grpname, sub in p.groupby("grp"):
            rows.append({"model": tag, "group": grpname, "n_days": sub.date.nunique(),
                         "WAPE": wape(sub.y, sub.pred)})
        # 每模型最差5日
        d = p.groupby("date").apply(lambda s: wape(s.y, s.pred)).sort_values(ascending=False)
        worst = "、".join(f"{dt.date()} {w:.1f}%" for dt, w in d.head(5).items())
        print(f"  {tag} 最差5日：{worst}")
    res = pd.DataFrame(rows)
    order = [g for g in order if g in set(res.group)]  # test期不存在的日型（如補班六）不列
    piv = res.pivot(index="group", columns="model", values="WAPE").reindex(order)
    piv["n_days"] = res.drop_duplicates("group").set_index("group").n_days.reindex(order)
    print("\n  按日型WAPE%：")
    print(piv.round(2).to_string())
    res.to_csv(REPORTS / "stage8_event_group_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(order))
    for i, tag in enumerate(["SARIMAX", "LightGBM", "LSTM"]):
        ax.bar(x + (i - 1) * 0.26, piv[tag], 0.26, label=tag)
    ax.set_xticks(x, [f"{g}\n(n={int(piv.n_days[g])})" for g in order], fontsize=9)
    ax.set_ylabel("WAPE (%)")
    ax.set_title("fig9　test期各日型誤差：統計 vs ML vs DL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9_test_event_errors.png", dpi=150)
    plt.close(fig)

    # SARIMAX最差5站（事件型場站）的三世代對照
    worst5 = ["國父紀念館", "動物園", "南港展覽館", "圓山", "淡水"]
    rows = []
    for name in worst5:
        for tag, p in preds.items():
            st = find_station(p, name)
            sub = p[p.station == st]
            rows.append({"station": name, "model": tag, "WAPE": wape(sub.y, sub.pred)})
    ws = pd.DataFrame(rows).pivot(index="station", columns="model", values="WAPE").loc[worst5]
    print("\n  SARIMAX最差5站（事件型場站）三世代對照 WAPE%：")
    print(ws.round(2).to_string())
    ws.to_csv(REPORTS / "stage8_worst_stations.csv")


# ================================================================ C. hold-out事件實驗
def part_c_holdout(df, cl_cols, params):
    print("\n========== C. Hold-out事件實驗（研究問題b）==========")
    feats_full = CAL_FEATS + LAG_FEATS + NET_FEATS + OTHER_STATIC + cl_cols
    feats_noty = [f for f in feats_full if f != "is_typhoon"]
    out = []

    # ---- C1 颱風：訓練截至事件前一日，有/無 is_typhoon
    typhoons = [("卡努", ["2023-08-03"], 0), ("凱米", ["2024-07-24", "2024-07-25"], 1),
                ("山陀兒", ["2024-10-02", "2024-10-03"], 2), ("康芮", ["2024-10-31"], 3)]
    print("  C1 颱風停班日（WAPE%，訓練嚴格<事件首日）：")
    for name, days, seen in typhoons:
        days = pd.to_datetime(days)
        base = df[df.date < days.min()]
        tgt = df[df.date.isin(days)]
        if len(base) < 5000 or len(tgt) == 0:
            continue
        res = {"event": f"颱風{name}", "dates": "/".join(d.strftime("%Y-%m-%d") for d in days),
               "n_prior_events": seen,
               "naive_lag1": wape(tgt.entries, tgt.lag1),
               "seasonal_lag7": wape(tgt.entries, tgt.lag7)}
        for tag, fts in [("with_flag", feats_full), ("no_flag", feats_noty)]:
            m = fit_lgbm(base[fts].values.astype(float), base.entries.values.astype(float), params)
            res[tag] = wape(tgt.entries, np.maximum(m.predict(tgt[fts].values.astype(float)), 0))
        res["flag_gain_pp"] = res["no_flag"] - res["with_flag"]
        out.append(res)
        print(f"    {name}（此前看過{seen}次颱風）: 有旗標 {res['with_flag']:.1f} ｜"
              f"無旗標 {res['no_flag']:.1f} ｜挽回 {res['flag_gain_pp']:+.1f}pp ｜"
              f"seasonal naive {res['seasonal_lag7']:.1f}")

    # ---- C2 新北耶誕城開幕（無特徵可編碼 → 量化資訊缺口）
    cutoff = pd.Timestamp("2025-11-14")
    base = df[df.date < cutoff]
    m = fit_lgbm(base[feats_full].values.astype(float), base.entries.values.astype(float), params)
    st = find_station(df, "BL板橋")
    tgt = df[(df.station == st) & (df.date >= cutoff) & (df.date <= "2025-11-16")]
    p = np.maximum(m.predict(tgt[feats_full].values.astype(float)), 0)
    print("\n  C2 新北耶誕城開幕（BL板橋，模型無任何事件特徵可用）：")
    for (_, r), pi in zip(tgt.iterrows(), p):
        gap = (r.entries - pi) / r.entries * 100
        print(f"    {r.date.date()}：實際 {r.entries:,.0f} ｜預測 {pi:,.0f} ｜低估 {gap:.1f}%")
        out.append({"event": "耶誕城開幕", "dates": str(r.date.date()),
                    "actual": r.entries, "pred_with_flag": pi, "underest_pct": gap})

    # ---- C3 跨年2025-12-31：有/無 is_nye（已看過2次跨年）
    cutoff = pd.Timestamp("2025-12-31")
    base = df[df.date < cutoff]
    tgt = df[df.date == cutoff]
    feats_nonye = [f for f in feats_full if f != "is_nye"]
    res = {"event": "跨年夜", "dates": "2025-12-31", "n_prior_events": 2,
           "seasonal_lag7": wape(tgt.entries, tgt.lag7)}
    for tag, fts in [("with_flag", feats_full), ("no_flag", feats_nonye)]:
        m = fit_lgbm(base[fts].values.astype(float), base.entries.values.astype(float), params)
        res[tag] = wape(tgt.entries, np.maximum(m.predict(tgt[fts].values.astype(float)), 0))
    res["flag_gain_pp"] = res["no_flag"] - res["with_flag"]
    out.append(res)
    print(f"\n  C3 跨年夜2025-12-31（系統級）：有is_nye {res['with_flag']:.1f}% ｜"
          f"無 {res['no_flag']:.1f}% ｜挽回 {res['flag_gain_pp']:+.1f}pp ｜"
          f"seasonal naive {res['seasonal_lag7']:.1f}%")

    pd.DataFrame(out).to_csv(REPORTS / "stage8_holdout_events.csv", index=False)

    # fig10（報告內文編號為「圖 8」）：颱風+跨年 旗標挽回效果
    # 藍柱上方標註「相對降幅」=(無旗標−有旗標)/無旗標——真正的訊號是
    # 「0次→≥1次的門檻」與「一致的七八成相對降幅」，不是絕對pp（各事件WAPE分母不同、pp不可比）
    ev = [r for r in out if "with_flag" in r and "no_flag" in r and r["event"] != "跨年夜"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(ev))
    labels = [f"{r['event']}\n(看過{r['n_prior_events']}次)" for r in ev]
    ax.bar(x - 0.28, [r["seasonal_lag7"] for r in ev], 0.26, label="seasonal naive", color="#bbb")
    ax.bar(x, [r["no_flag"] for r in ev], 0.26, label="LGBM 無事件旗標", color="#c0504d")
    bars = ax.bar(x + 0.28, [r["with_flag"] for r in ev], 0.26, label="LGBM 有事件旗標", color="#4878a8")
    for b, r in zip(bars, ev):
        rel = (r["no_flag"] - r["with_flag"]) / r["no_flag"] * 100 if r["no_flag"] else 0.0
        ax.annotate(f"−{rel:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#2b4d6f", fontweight="bold")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("事件日WAPE (%)")
    ax.set_title("事件旗標的挽回效果（hold-out：訓練嚴格早於事件）\n藍柱上標註＝相對降幅 (無旗標−有旗標)/無旗標")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10_holdout_events.png", dpi=150)
    plt.close(fig)

    print("\n  C4 0403地震：不做retrain。service_suspended於2024-04-04前全為0，")
    print("     樹模型對訓練期未見過的旗標值無法建立切分（無法外推）——此為樹模型的")
    print("     結構限制，直接寫入報告；事後期間（旗標已見過）模型能正確預測近0值。")


def main(skip_shap=False, skip_holdout=False):
    np.random.seed(SEED)
    df, cl_cols = load_data()
    params = best_params()
    print(f"LightGBM params（自stage6 tuning log）：{params}")
    if not skip_shap:
        part_a_shap(df, cl_cols, params)
    part_b_events(df)
    if not skip_holdout:
        part_c_holdout(df, cl_cols, params)
    print("\n完成。產出：stage8_shap_importance / event_group_errors / worst_stations / "
          "holdout_events .csv + figures/fig6~fig10.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--skip-holdout", action="store_true")
    a = ap.parse_args()
    main(skip_shap=a.skip_shap, skip_holdout=a.skip_holdout)
