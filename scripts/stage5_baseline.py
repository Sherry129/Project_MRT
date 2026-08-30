# -*- coding: utf-8 -*-
"""
階段5：Baseline模型 — 統計基準線 + SARIMAX
================================
輸入：data/features/feature_table.csv（stage4產出）
輸出：reports/stage5_baseline_metrics.csv（基準線）
     reports/stage5_sarima_metrics.csv（SARIMAX逐站）

評估設定（與後續ML/DL模型一致，確保公平比較）：
- 切分：train+val = 2023-01-29 ~ 2025-12-31；test = 2026-01-01 ~ 2026-05-31（151天）
- 一步預測（one-step-ahead）：預測第t日時可用t-1（含）之前的所有真實觀測
- 指標：MAE、RMSE、WAPE（=Σ|誤差|/Σ實際值，對大小站不偏袒）
- 分組報告：整體 + 站點功能分群（residential/employment/employment_peak/tourism_hub/mixed）

SARIMAX規格：
- 逐站擬合 SARIMA(1,0,1)(1,1,1)_7 + exog[is_offday, is_lny, is_nye, is_makeup_workday]
- 週季節差分d=7捕捉星期節律；exog吸收假日/春節/跨年效應
- test期間用 .extend() 做filtering（不重新估參，延續訓練期末狀態），即真正的一步預測

安裝需求：pip install pandas numpy statsmodels tqdm
用法：python scripts/stage5_baseline.py [--limit N]  # N=只跑前N站（測試用）
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "feature_table.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
TRAIN_END = "2025-12-31"
EXOG = ["is_offday", "is_lny", "is_nye", "is_makeup_workday"]


def metrics(y, p):
    e = np.asarray(y, float) - np.asarray(p, float)
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                WAPE=np.abs(e).sum() / np.asarray(y, float).sum() * 100)


def run_naive_baselines(df):
    tr = df[df.date <= TRAIN_END]
    te = df[df.date > TRAIN_END].reset_index(drop=True)
    preds = {"naive_lag1": te.lag1.values,
             "seasonal_naive_lag7": te.lag7.values,
             "rolling28_mean": te.roll28_mean.values}

    # 逐站OLS（dow dummies + 事件旗標 + lag/rolling）
    num_cols = ["lag1", "lag7", "lag14", "roll7_mean", "roll28_mean"]

    def design(d):
        X = [np.ones(len(d))]
        X += [(d.dow == k).astype(float).values for k in range(6)]
        X += [d[c].astype(float).values for c in EXOG]
        X += [d[c].values / 1e4 for c in num_cols]
        return np.column_stack(X)

    ols = np.full(len(te), np.nan)
    for s, tr_s in tr.groupby("station"):
        beta, *_ = np.linalg.lstsq(design(tr_s), tr_s.entries.values.astype(float), rcond=None)
        m = (te.station == s).values
        ols[m] = design(te[m]) @ beta
    preds["ols_per_station"] = np.maximum(ols, 0)

    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "group": "ALL", **metrics(te.entries, p)})
        for cl, sub in te.groupby("cluster"):
            rows.append({"model": name, "group": cl,
                         **metrics(sub.entries, p[sub.index.values])})
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "stage5_baseline_metrics.csv", index=False)
    print(res[res.group == "ALL"].round(1).to_string(index=False))
    return te


def run_sarimax(df, limit=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    stations = sorted(df.station.unique())[:limit]
    rows, preds_all = [], []
    for s in tqdm(stations, desc="SARIMAX"):
        d = df[df.station == s].sort_values("date")
        tr = d[d.date <= TRAIN_END]
        te = d[d.date > TRAIN_END]
        try:
            model = SARIMAX(tr.entries.values.astype(float),
                            exog=tr[EXOG].values.astype(float),
                            order=(1, 0, 1), seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200)
            # 一步預測：用已估參數filter測試期觀測
            # .extend() 延續訓練期結束時的Kalman filter狀態，才是「預測第t日時可用
            # t-1(含)之前所有真實觀測」這個協定的正確實作。
            # 舊版改用 .apply() 只帶14天暖身，會觸發diffuse initialization：季節差分
            # (D=1, s=7)先吃掉7個觀測，僅剩約7個差分點，不足以讓含季節AR/MA的狀態向量
            # 收斂。實測前7日WAPE 14.68%（vs 5.92%），且污染延續至第15日之後
            # （4.86% vs 4.11%），站均WAPE整體高估約1.4pp。
            # 註：.extend() 與「.apply()全序列」實測逐站結果完全相同（最大差異0.0）。
            ext = fit.extend(endog=te.entries.values.astype(float),
                             exog=te[EXOG].values.astype(float))
            p = np.maximum(ext.get_prediction().predicted_mean, 0)
            m = metrics(te.entries.values, p)
            rows.append({"station": s, "cluster": d.cluster.iloc[0], **m,
                         "aic": fit.aic, "converged": fit.mle_retvals.get("converged", True)})
            preds_all.append(pd.DataFrame({"station": s, "date": te.date.values,
                                           "y": te.entries.values, "pred": p}))
        except Exception as e:
            rows.append({"station": s, "cluster": d.cluster.iloc[0],
                         "MAE": np.nan, "RMSE": np.nan, "WAPE": np.nan,
                         "aic": np.nan, "converged": False})
            print(f"  {s} 失敗: {e}")
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "stage5_sarima_metrics.csv", index=False)
    if preds_all:
        pd.concat(preds_all).to_csv(REPORTS / "stage5_sarima_predictions.csv", index=False)
    ok = res.dropna(subset=["WAPE"])
    print(f"\nSARIMAX 完成 {len(ok)}/{len(res)} 站")
    print("整體 WAPE(加權): %.2f%%" % (
        sum(r.MAE * 1 for _, r in ok.iterrows()) and
        (pd.concat(preds_all).pipe(lambda d: (d.y - d.pred).abs().sum() / d.y.sum() * 100))))
    print("各站型平均 WAPE:")
    print(ok.groupby("cluster").WAPE.mean().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前N站（快速測試）")
    args = ap.parse_args()

    df = pd.read_csv(FEAT, parse_dates=["date"])
    df = df.dropna(subset=["lag1", "lag7", "lag14", "roll7_mean", "roll28_mean"])
    print("=== 統計基準線 ===")
    run_naive_baselines(df)
    print("\n=== SARIMAX 逐站（119站約10-30分鐘）===")
    run_sarimax(df, limit=args.limit)
