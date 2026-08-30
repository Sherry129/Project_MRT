# -*- coding: utf-8 -*-
"""
階段6b：分群粒度消融實驗（k=3 vs k=4 vs 無分群）
=================================================
目的：Stage 2 的 k 值選擇缺乏客觀依據（elbow指向k=3、silhouette指向k=2、gap無停止點）。
      本實驗改由下游預測效能裁決：分群標籤的粒度到底影響模型多少？

設計（與 stage6_ml.py 同協定）：
- 切分：train <= 2024-12-31 / val 2025 / test 2026-01~05
- 模型：LightGBM，train調參+val early stopping，再以 train+val 重訓、test評估
- 三個arm只差在 cluster one-hot：no_cluster / k3(3欄) / k4(4欄)，其餘特徵完全相同
- 關鍵控制：跑「完整超參數矩陣」而非各arm各自調參。各自調參會讓超參數噪音
  混進arm差異裡，本實驗證實那正是假訊號的來源。
- 不確定度：對test日期做區塊bootstrap，給差異的95%信賴區間

輸入：data/parquet/od/od_*.parquet、data/features/feature_table.csv
輸出：reports/stage6b_k_ablation_matrix.csv、reports/stage6b_k_ablation_boot.csv
用法：python scripts/stage6b_k_ablation.py
"""
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PQ = (ROOT / "data" / "parquet" / "od" / "od_*.parquet").as_posix()
FEAT = ROOT / "data" / "features" / "feature_table.csv"
REPORTS = ROOT / "reports"

SEED = 42
TRAIN_END = "2024-12-31"
VAL_END = "2025-12-31"
GRID = [(nl, mc) for nl in (63, 127, 255) for mc in (20, 50)]

CAL_FEATS = ["dow", "month", "is_weekend", "is_holiday", "is_offday",
             "is_makeup_workday", "is_lny", "is_typhoon", "is_nye"]
LAG_FEATS = ["lag1", "lag7", "lag14", "roll7_mean", "roll28_mean"]
NET_FEATS = ["pagerank", "out_strength_train", "in_strength_train", "deg_out", "deg_in"]
OTHER_STATIC = ["is_yline", "service_suspended"]


def wape(y, p):
    y = np.asarray(y, float)
    return np.abs(y - np.asarray(p, float)).sum() / y.sum() * 100


def kmeans(X, K, seeds=8, iters=200):
    best = None
    for s in range(seeds):
        r = np.random.default_rng(s)
        C = X[r.choice(len(X), K, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
            if np.allclose(C2, C):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1], best[0]


def build_labels():
    """重建 k=3 / k=4 分群標籤，僅用訓練期資料避免洩漏。

    注意分母：必須除以「訓練窗內該日型的總天數」，不可用
    SUM(trips)/COUNT(DISTINCT date)。後者在深夜稀疏時段只會除以少數幾天
    （例如02時僅跨年延長營運當日有紀錄），把跨年人潮放大成假的日常行為訊號，
    進而在k=4時捏造出一個由信義線東段站點組成的偽群。stage2/stage4 目前的
    聚合SQL即有此問題，見 reports/stage6b_k_ablation.md。
    """
    con = duckdb.connect()
    prof = con.execute(f"""
        WITH d AS (SELECT DISTINCT date,
                          CASE WHEN dayofweek(date) IN (0,6) THEN 1 ELSE 0 END wend
                   FROM read_parquet('{PQ}') WHERE date <= '{TRAIN_END}'),
             nd AS (SELECT wend, COUNT(*) n FROM d GROUP BY 1)
        SELECT b.origin AS station, b.hour,
               CASE WHEN dayofweek(b.date) IN (0,6) THEN 1 ELSE 0 END AS wend,
               SUM(b.trips)::DOUBLE / MAX(nd.n) AS avg_trips
        FROM read_parquet('{PQ}') b
        JOIN nd ON nd.wend = CASE WHEN dayofweek(b.date) IN (0,6) THEN 1 ELSE 0 END
        WHERE b.date <= '{TRAIN_END}' GROUP BY 1, 2, 3
    """).fetchdf()
    con.close()

    sts = sorted(prof.station.unique())
    pv = prof.pivot_table(index="station", columns=["wend", "hour"],
                          values="avg_trips", fill_value=0).reindex(sts).fillna(0)
    X = pv.div(pv.sum(1).replace(0, 1), axis=0).values
    wkn = pv[0].values / pv[0].values.sum(1, keepdims=True)

    out = pd.DataFrame({"station": sts})
    for K in (3, 4):
        lab, inertia = kmeans(X, K)
        out[f"cluster_k{K}"] = [f"c{l}" for l in lab]
        print(f"\n=== k={K}  inertia={inertia:.5f} ===")
        for k in range(K):
            m = wkn[lab == k].mean(0)
            big = pv.index[lab == k][np.argsort(-pv.values[lab == k].sum(1))][:4].tolist()
            print(f"  c{k} n={(lab == k).sum():3d} 量占比={pv.values[lab == k].sum() / pv.values.sum() * 100:5.1f}%"
                  f"  早峰(7-9)={m[7:10].sum() * 100:4.1f}%  晚峰(17-19)={m[17:20].sum() * 100:4.1f}%"
                  f"  峰值{m.max() * 100:.1f}%@{m.argmax()}時 | {'、'.join(big)}")
    return out


def load(labels):
    df = pd.read_csv(FEAT, parse_dates=["date"]).dropna(subset=LAG_FEATS).reset_index(drop=True)
    df = df.merge(labels, on="station", how="left")
    assert df[["cluster_k3", "cluster_k4"]].notna().all().all(), "有站點缺分群標籤"
    for K in (3, 4):
        df = pd.concat([df, pd.get_dummies(df[f"cluster_k{K}"], prefix=f"k{K}").astype(int)], axis=1)
    arms = {"no_cluster": [],
            "k3": [c for c in df.columns if c.startswith("k3_")],
            "k4": [c for c in df.columns if c.startswith("k4_")]}
    return df, arms


def run_matrix(df, arms):
    """3個arm × 6組超參數的完整矩陣。回傳(結果表, 各arm最佳設定下的test預測)。"""
    import lightgbm as lgb
    tr = df[df.date <= TRAIN_END]
    va = df[(df.date > TRAIN_END) & (df.date <= VAL_END)]
    te = df[df.date > VAL_END].reset_index(drop=True)
    yte = te.entries.values.astype(float)
    rows, preds = [], {}

    for arm, cols in arms.items():
        f = CAL_FEATS + LAG_FEATS + NET_FEATS + OTHER_STATIC + cols
        Xtr, ytr = tr[f].values.astype(float), tr.entries.values.astype(float)
        Xva, yva = va[f].values.astype(float), va.entries.values.astype(float)
        Xte = te[f].values.astype(float)
        Xfull, yfull = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
        best = None
        for nl, mc in GRID:
            P = dict(num_leaves=nl, min_child_samples=mc, learning_rate=0.05)
            m = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, n_estimators=3000, **P)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            n = int(m.best_iteration_ or 3000)
            vw = wape(yva, np.maximum(m.predict(Xva), 0))
            m2 = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, n_estimators=n, **P)
            m2.fit(Xfull, yfull)
            p = np.maximum(m2.predict(Xte), 0)
            rows.append(dict(arm=arm, num_leaves=nl, min_child=mc, n_est=n,
                             val_WAPE=vw, test_WAPE=wape(yte, p)))
            print(f"{arm:11s} nl={nl:3d} mc={mc:2d} -> val {vw:.4f}%  test {rows[-1]['test_WAPE']:.4f}%")
            if best is None or vw < best[0]:
                best = (vw, p)
        preds[arm] = best[1]
    return pd.DataFrame(rows), preds, te, yte


def bootstrap_diff(te, yte, preds, B=2000, seed=0):
    """對test日期做區塊bootstrap，估arm間WAPE差異的95%信賴區間。"""
    rng = np.random.default_rng(seed)
    dates = te.date.values
    ud = np.unique(dates)
    didx = {d: np.where(dates == d)[0] for d in ud}
    diffs = {k: [] for k in ["k4-k3", "k4-no_cluster", "k3-no_cluster"]}
    for _ in range(B):
        ii = np.concatenate([didx[d] for d in rng.choice(ud, len(ud), replace=True)])
        y = yte[ii]
        w = {a: np.abs(y - p[ii]).sum() / y.sum() * 100 for a, p in preds.items()}
        diffs["k4-k3"].append(w["k4"] - w["k3"])
        diffs["k4-no_cluster"].append(w["k4"] - w["no_cluster"])
        diffs["k3-no_cluster"].append(w["k3"] - w["no_cluster"])
    out = []
    for k, d in diffs.items():
        d = np.array(d)
        out.append(dict(comparison=k, point=d.mean(), lo95=np.percentile(d, 2.5),
                        hi95=np.percentile(d, 97.5), p_improve=(d < 0).mean()))
        print(f"{k:16s} 點估計={d.mean():+.4f}pp  95%CI=[{out[-1]['lo95']:+.4f}, {out[-1]['hi95']:+.4f}]"
              f"  P(有改善)={out[-1]['p_improve'] * 100:.1f}%")
    return pd.DataFrame(out)


def main():
    labels = build_labels()
    df, arms = load(labels)
    print(f"\none-hot欄位：" + "  ".join(f"{a}={len(c)}欄" for a, c in arms.items()))
    R, preds, te, yte = run_matrix(df, arms)
    R.to_csv(REPORTS / "stage6b_k_ablation_matrix.csv", index=False)

    print("\n=== test WAPE 矩陣 ===")
    print(R.pivot_table(index=["num_leaves", "min_child"], columns="arm", values="test_WAPE").round(4).to_string())
    print("\n=== 每arm的 test WAPE 分布（跨6組超參數）===")
    print(R.groupby("arm").test_WAPE.agg(["mean", "std", "min", "max"]).round(4).to_string())
    span = (R.groupby("arm").test_WAPE.max() - R.groupby("arm").test_WAPE.min()).mean()
    mu = R.groupby("arm").test_WAPE.mean()
    print(f"\n超參數造成的變異（arm內 max-min 平均）= {span:.4f} pp")
    print(f"分群粒度造成的變異（k4-k3 平均）      = {mu['k4'] - mu['k3']:+.4f} pp")
    print(f"→ 訊噪比 {abs(mu['k4'] - mu['k3']) / span:.3f}：分群粒度的影響遠低於調參噪音")

    print("\n=== 分群體 test WAPE（以k=4標籤分組）===")
    g = te.cluster_k4.values
    vol = pd.Series(yte).groupby(g).sum() / yte.sum() * 100
    print(f"{'group':6s}{'量占比':>8s}" + "".join(f"{a:>13s}" for a in preds))
    for grp in sorted(set(g)):
        i = g == grp
        print(f"{grp:6s}{vol[grp]:7.1f}%" + "".join(f"{wape(yte[i], p[i]):12.4f}%" for p in preds.values()))

    print("\n=== 日期區塊 bootstrap（B=2000）===")
    bootstrap_diff(te, yte, preds).to_csv(REPORTS / "stage6b_k_ablation_boot.csv", index=False)
    print(f"\n已輸出至 {REPORTS}")


if __name__ == "__main__":
    main()
