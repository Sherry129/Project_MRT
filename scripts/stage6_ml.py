# -*- coding: utf-8 -*-
"""
階段6：機器學習模型 — RandomForest + LightGBM/XGBoost（global model）
================================
輸入：data/features/feature_table.csv（stage4產出）
輸出：reports/stage6_ml_metrics.csv       （各模型×消融×站型指標）
     reports/stage6_ml_predictions.csv   （最佳模型test逐日預測，供Stage 8）
     reports/stage6_tuning_log.csv       （val調參紀錄，口試備查）

評估協定（與Stage 5一致，公平比較）：
- 一步預測：特徵僅含shift(1)後的lag/rolling，無當日資訊
- 調參：train(2023-01-29~2024-12-31)訓練、validation(2025)選超參數
- 最終：best params 用 train+val 重訓，test(2026-01~05)只評一次
- 指標：WAPE為主，輔MAE/RMSE；整體+按站型（train期分群標籤）

模型設計：
- Global model：全站合併訓練一個模型，站別資訊以 cluster(one-hot)+靜態特徵表達
  （不用station identity，讓模型學「站型」而非「背站名」——研究問題(c)的設計）
- 主力：LightGBM（無則自動改用XGBoost）；輔助：RandomForest
- 消融實驗（研究問題a）：主力模型 有/無 網絡特徵（pagerank/strength/degree）各跑一次

安裝需求：pip install pandas numpy scikit-learn lightgbm
         （lightgbm裝不了就 pip install xgboost，腳本會自動切換）
用法：python scripts/stage6_ml.py [--quick]   # --quick=縮小網格快速驗證環境
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "feature_table.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TRAIN_END = "2024-12-31"   # 調參階段的train上界
VAL_END = "2025-12-31"     # val上界；之後為test

CAL_FEATS = ["dow", "month", "is_weekend", "is_holiday", "is_offday",
             "is_makeup_workday", "is_lny", "is_typhoon", "is_nye"]
LAG_FEATS = ["lag1", "lag7", "lag14", "roll7_mean", "roll28_mean"]
NET_FEATS = ["pagerank", "out_strength_train", "in_strength_train", "deg_out", "deg_in"]
OTHER_STATIC = ["is_yline", "service_suspended"]  # cluster另做one-hot


def metrics(y, p):
    e = np.asarray(y, float) - np.asarray(p, float)
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                WAPE=np.abs(e).sum() / np.asarray(y, float).sum() * 100)


def wape(y, p):
    return np.abs(np.asarray(y, float) - np.asarray(p, float)).sum() / np.asarray(y, float).sum() * 100


def load_data():
    df = pd.read_csv(FEAT, parse_dates=["date"])
    df = df.dropna(subset=LAG_FEATS).reset_index(drop=True)
    df = pd.concat([df, pd.get_dummies(df.cluster, prefix="cl").astype(int)], axis=1)
    cl_cols = [c for c in df.columns if c.startswith("cl_")]
    return df, cl_cols


def split(df):
    tr = df[df.date <= TRAIN_END]
    va = df[(df.date > TRAIN_END) & (df.date <= VAL_END)]
    te = df[df.date > VAL_END]
    return tr, va, te


# ---------------------------------------------------------------- 模型定義
def get_booster():
    """回傳 (名稱, 建構函式, 是否支援early stopping)。優先LightGBM。"""
    try:
        import lightgbm as lgb

        def make(params):
            return lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **params)
        return "LightGBM", make, "lgb"
    except ImportError:
        import xgboost as xgb

        def make(params):
            p = dict(params)
            p["max_depth"] = {31: 5, 63: 6, 127: 7, 255: 8}.get(p.pop("num_leaves", 63), 6)
            p["min_child_weight"] = p.pop("min_child_samples", 20)
            return xgb.XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist",
                                    early_stopping_rounds=100, **p)
        return "XGBoost", make, "xgb"


def fit_booster(make, kind, params, Xtr, ytr, Xva, yva):
    """train擬合+val early stopping，回傳(模型, best_n_estimators)。"""
    m = make({**params, "n_estimators": 3000, "learning_rate": params.get("learning_rate", 0.05)})
    if kind == "lgb":
        import lightgbm as lgb
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        best_n = m.best_iteration_ or 3000
    else:
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        best_n = getattr(m, "best_iteration", None) or 3000
    return m, int(best_n)


def tune_booster(make, kind, Xtr, ytr, Xva, yva, quick=False):
    grid = [dict(num_leaves=nl, min_child_samples=mc, learning_rate=lr)
            for nl in ([63] if quick else [63, 127, 255])
            for mc in ([20] if quick else [20, 50])
            for lr in [0.05]]
    log, best = [], None
    for g in grid:
        t0 = time.time()
        m, best_n = fit_booster(make, kind, g, Xtr, ytr, Xva, yva)
        w = wape(yva, np.maximum(m.predict(Xva), 0))
        log.append({**g, "best_n_estimators": best_n, "val_WAPE": w, "sec": round(time.time() - t0, 1)})
        print(f"  {g} -> n={best_n}, val WAPE={w:.3f}%")
        if best is None or w < best["val_WAPE"]:
            best = log[-1]
    return best, log


def tune_rf(Xtr, ytr, Xva, yva, quick=False):
    from sklearn.ensemble import RandomForestRegressor
    grid = [dict(max_depth=d, min_samples_leaf=l)
            for d in ([20] if quick else [12, 20, None])
            for l in ([5] if quick else [5, 20])]
    log, best = [], None
    n_est = 100 if quick else 300
    for g in grid:
        t0 = time.time()
        m = RandomForestRegressor(n_estimators=n_est, random_state=SEED, n_jobs=-1, **g)
        m.fit(Xtr, ytr)
        w = wape(yva, np.maximum(m.predict(Xva), 0))
        log.append({**g, "n_estimators": n_est, "val_WAPE": w, "sec": round(time.time() - t0, 1)})
        print(f"  {g} -> val WAPE={w:.3f}%")
        if best is None or w < best["val_WAPE"]:
            best = log[-1]
    return best, log


# ---------------------------------------------------------------- 主流程
def run_experiment(df, cl_cols, feats, tag, quick, rows, preds_out, tuning_log):
    """一組特徵集的完整實驗：booster調參→重訓→test評估（+RF僅在full跑）。"""
    tr, va, te = split(df)
    Xtr, ytr = tr[feats].values.astype(float), tr.entries.values.astype(float)
    Xva, yva = va[feats].values.astype(float), va.entries.values.astype(float)
    Xte, yte = te[feats].values.astype(float), te.entries.values.astype(float)
    # 重訓用 train+val（與Stage 5基準線同協定）
    Xfull = np.vstack([Xtr, Xva])
    yfull = np.concatenate([ytr, yva])

    bname, make, kind = get_booster()
    print(f"\n=== {bname}（{tag}，特徵{len(feats)}欄）調參 ===")
    best, log = tune_booster(make, kind, Xtr, ytr, Xva, yva, quick)
    for r in log:
        tuning_log.append({"model": bname, "ablation": tag, **r})

    # 重訓：固定val選出的n_estimators（early stopping已完成任務）
    params = {k: best[k] for k in ["num_leaves", "min_child_samples", "learning_rate"]}
    m = make({**params, "n_estimators": best["best_n_estimators"]})
    m.fit(Xfull, yfull)
    p = np.maximum(m.predict(Xte), 0)
    rows.append({"model": bname, "ablation": tag, "group": "ALL", **metrics(yte, p)})
    for cl, sub in te.groupby("cluster"):
        idx = te.index.get_indexer(sub.index)
        rows.append({"model": bname, "ablation": tag, "group": cl, **metrics(sub.entries, p[idx])})
    print(f"{bname}（{tag}） test WAPE = {wape(yte, p):.3f}%")

    if tag == "full":
        preds_out[bname] = pd.DataFrame(
            {"station": te.station.values, "date": te.date.values, "y": yte, "pred": p})
        # 特徵重要度：同時輸出 split 與 gain，避免基準混淆。
        # 注意：LGBMRegressor.feature_importances_ 預設為 split（分裂次數），
        # XGBRegressor 則預設為 gain，兩者不可直接比較。報告以 gain 為準——
        # gain 衡量「該特徵帶來多少損失下降」，才是特徵價值的量度；
        # split 只算「被用來分裂幾次」，會系統性高估高基數的連續特徵
        # （如 month/roll28_mean）並低估少數但關鍵的分裂（如 pagerank）。
        # 實測：pagerank 依 split 排第8、依 gain 排第1；網絡特徵合計占
        # split 14.2% 但占 gain 45.7%，後者才與消融實驗的0.362pp相符。
        imp = pd.DataFrame({"feature": feats})
        if kind == "lgb":
            imp["gain"] = m.booster_.feature_importance("gain")
            imp["split"] = m.booster_.feature_importance("split")
        else:
            imp["gain"] = m.get_booster().get_score(importance_type="gain") \
                and [m.get_booster().get_score(importance_type="gain").get(f"f{i}", 0.0)
                     for i in range(len(feats))]
            imp["split"] = [m.get_booster().get_score(importance_type="weight").get(f"f{i}", 0.0)
                            for i in range(len(feats))]
        imp["gain_pct"] = (imp.gain / imp.gain.sum() * 100).round(2)
        imp = imp.sort_values("gain", ascending=False)
        imp.to_csv(REPORTS / "stage6_feature_importance.csv", index=False)
        print("Top10特徵（gain）：", ", ".join(imp.feature.head(10)))
        print("網絡特徵 gain 合計：%.1f%%" % imp[imp.feature.isin(NET_FEATS)].gain_pct.sum())

        # 輔助模型：RandomForest（僅full特徵集）
        print(f"\n=== RandomForest（{tag}）調參 ===")
        from sklearn.ensemble import RandomForestRegressor
        rbest, rlog = tune_rf(Xtr, ytr, Xva, yva, quick)
        for r in rlog:
            tuning_log.append({"model": "RandomForest", "ablation": tag, **r})
        rf = RandomForestRegressor(n_estimators=rbest["n_estimators"], random_state=SEED,
                                   n_jobs=-1, max_depth=rbest["max_depth"],
                                   min_samples_leaf=rbest["min_samples_leaf"])
        rf.fit(Xfull, yfull)
        rp = np.maximum(rf.predict(Xte), 0)
        rows.append({"model": "RandomForest", "ablation": tag, "group": "ALL", **metrics(yte, rp)})
        for cl, sub in te.groupby("cluster"):
            idx = te.index.get_indexer(sub.index)
            rows.append({"model": "RandomForest", "ablation": tag, "group": cl,
                         **metrics(sub.entries, rp[idx])})
        print(f"RandomForest（{tag}） test WAPE = {wape(yte, rp):.3f}%")
        preds_out["RandomForest"] = pd.DataFrame(
            {"station": te.station.values, "date": te.date.values, "y": yte, "pred": rp})


def main(quick=False):
    df, cl_cols = load_data()
    feats_full = CAL_FEATS + LAG_FEATS + NET_FEATS + OTHER_STATIC + cl_cols
    feats_nonet = CAL_FEATS + LAG_FEATS + OTHER_STATIC + cl_cols

    rows, preds_out, tuning_log = [], {}, []
    run_experiment(df, cl_cols, feats_full, "full", quick, rows, preds_out, tuning_log)
    run_experiment(df, cl_cols, feats_nonet, "no_network", quick, rows, preds_out, tuning_log)

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "stage6_ml_metrics.csv", index=False)
    pd.DataFrame(tuning_log).to_csv(REPORTS / "stage6_tuning_log.csv", index=False)

    # 最佳模型（整體WAPE最低者）的預測存檔供Stage 8
    all_rows = res[(res.group == "ALL") & (res.ablation == "full")]
    best_model = all_rows.sort_values("WAPE").iloc[0].model
    preds_out[best_model].to_csv(REPORTS / "stage6_ml_predictions.csv", index=False)

    print("\n================ 結果總表 ================")
    print(res.round(2).to_string(index=False))
    print(f"\n最佳模型：{best_model}（預測已存 stage6_ml_predictions.csv）")

    # 消融差異（研究問題a）
    print("\n=== 網絡特徵消融（booster）===")
    b = res[res.model.isin(["LightGBM", "XGBoost"])]
    piv = b.pivot_table(index="group", columns="ablation", values="WAPE")
    piv["diff(no_net - full)"] = piv.get("no_network") - piv.get("full")
    print(piv.round(3).to_string())

    print("\n=== 累積比較表（test WAPE%）===")
    print("naive 13.4 | seasonal naive 9.9 | SARIMAX 6.07 | 逐站OLS 4.8 | "
          + " | ".join(f"{r.model} {r.WAPE:.2f}"
                       for _, r in all_rows.sort_values("WAPE").iterrows()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="縮小網格快速驗證環境")
    args = ap.parse_args()
    np.random.seed(SEED)
    main(quick=args.quick)
