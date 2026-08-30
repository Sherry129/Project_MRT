# -*- coding: utf-8 -*-
"""
階段7：深度學習模型 — LSTM global model（PyTorch）
================================
輸入：data/features/feature_table.csv（stage4產出）
輸出：reports/stage7_dl_metrics.csv       （整體+站型指標）
     reports/stage7_dl_predictions.csv   （test逐日預測，供Stage 8）
     reports/stage7_training_log.csv     （各配置訓練曲線，口試備查）

評估協定（與Stage 5/6完全一致）：
- 一步預測：輸入序列 = 目標日前28天的「真實」進站量（自然滿足one-step-ahead）
- 調參：train(2023-01-29~2024-12-31)訓練、validation(2025)早停+選配置
- 最終：best配置以train+val重訓固定epoch數，test只評一次
- 指標：WAPE為主，輔MAE/RMSE；整體+按站型

模型設計：
- Global model + station embedding（依handoff §6規劃；與Stage 6「無站名」設計互補——
  報告需說明差異：LSTM允許認站，回答「認站的DL vs 不認站的ML」誰強）
- 輸入三部分：
  (a) 序列：前28天 [z-score進站量, is_offday, is_holiday]（每站z-score用train期均值/標準差，防洩漏）
  (b) 目標日情境：dow one-hot、月份sin/cos、假日/補班/春節/颱風/跨年/停駛旗標、
      cluster one-hot、標準化靜態特徵（pagerank、log流量強度、度數）、is_yline
  (c) station embedding（可學習向量）
- LSTM末隱狀態 ⊕ 情境 ⊕ embedding → MLP → 預測z-score → 反轉換回人次
- 損失：Huber（z-score尺度）；固定seed；early stopping用val WAPE（原始尺度）

安裝需求：pip install torch --index-url https://download.pytorch.org/whl/cpu
         （有NVIDIA GPU可裝CUDA版，腳本自動偵測）
用法：python scripts/stage7_dl.py [--quick]   # --quick=單配置3epoch驗證環境
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
SEQ_LEN = 28

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "feature_table.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TRAIN_END = "2024-12-31"
VAL_END = "2025-12-31"

FLAGS = ["is_offday", "is_holiday", "is_makeup_workday", "is_lny",
         "is_typhoon", "is_nye", "service_suspended"]
STATIC_NUM = ["pagerank", "out_strength_train", "in_strength_train", "deg_out", "deg_in"]


def metrics(y, p):
    e = np.asarray(y, float) - np.asarray(p, float)
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                WAPE=np.abs(e).sum() / np.asarray(y, float).sum() * 100)


def wape(y, p):
    return metrics(y, p)["WAPE"]


# ---------------------------------------------------------------- 資料構建（純numpy，可在無torch環境測試）
def build_dataset():
    df = pd.read_csv(FEAT, parse_dates=["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    stations = sorted(df.station.unique())
    sid_map = {s: i for i, s in enumerate(stations)}
    n_days = df.date.nunique()
    assert len(df) == len(stations) * n_days, "feature_table應為完整站×日網格"

    # 每站z-score參數：僅用train期（防洩漏）
    tr_mask = df.date <= TRAIN_END
    stats = df[tr_mask].groupby("station").entries.agg(["mean", "std"])
    stats["std"] = stats["std"].clip(lower=1.0)
    df = df.merge(stats.rename(columns={"mean": "mu", "std": "sd"}), on="station")
    df["z"] = (df.entries - df.mu) / df.sd

    # 靜態特徵標準化（全站分布，靜態特徵本身已僅用train期OD算出）
    static = df.drop_duplicates("station").set_index("station").loc[stations]
    S = np.column_stack([static.pagerank.values,
                         np.log1p(static.out_strength_train.values),
                         np.log1p(static.in_strength_train.values),
                         static.deg_out.values, static.deg_in.values]).astype(np.float32)
    S = (S - S.mean(0)) / np.where(S.std(0) == 0, 1, S.std(0))
    cl_dummies = pd.get_dummies(static.cluster).astype(np.float32)
    S = np.hstack([S, cl_dummies.values, static.is_yline.values[:, None].astype(np.float32)])

    # 逐站矩陣（站×日）
    def pivot(col):
        return df.pivot(index="station", columns="date", values=col).loc[stations].values

    Z = pivot("z").astype(np.float32)
    OFF = pivot("is_offday").astype(np.float32)
    HOL = pivot("is_holiday").astype(np.float32)
    dates = np.sort(df.date.unique())

    # 目標日情境特徵（日級，全站共用部分）
    day = df.drop_duplicates("date").sort_values("date")
    dow_oh = np.eye(7, dtype=np.float32)[day.dow.values]
    m = day.month.values
    mon = np.column_stack([np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]).astype(np.float32)
    flags_by_day = day[["is_offday", "is_holiday", "is_makeup_workday",
                        "is_lny", "is_typhoon", "is_nye"]].values.astype(np.float32)
    CTX_DAY = np.hstack([dow_oh, mon, flags_by_day])          # (n_days, 15)
    SUSP = pivot("service_suspended").astype(np.float32)       # 站×日（唯一站×日交互旗標）

    # 樣本：目標日索引 t >= SEQ_LEN
    n_st = len(stations)
    t_idx = np.arange(SEQ_LEN, n_days)
    # 序列 (樣本, 28, 3)：滑動視窗
    def windows(M):
        # M: (站, 日) -> (站, len(t_idx), 28)
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(M, SEQ_LEN, axis=1)[:, :len(t_idx), :]
        return w
    Xseq = np.stack([windows(Z), windows(OFF), windows(HOL)], axis=-1)  # (站, T, 28, 3)
    Xseq = Xseq.reshape(-1, SEQ_LEN, 3)

    sid = np.repeat(np.arange(n_st), len(t_idx)).astype(np.int64)
    day_of = np.tile(t_idx, n_st)
    Xctx = np.hstack([CTX_DAY[day_of], SUSP.reshape(-1)[
        (np.repeat(np.arange(n_st), len(t_idx)) * n_days + day_of)][:, None],
        S[sid]]).astype(np.float32)
    y_raw = pivot("entries").astype(np.float32)[np.repeat(np.arange(n_st), len(t_idx)), day_of]
    mu = stats["mean"].loc[stations].values.astype(np.float32)[sid]
    sd = stats["std"].loc[stations].values.astype(np.float32)[sid]
    yz = (y_raw - mu) / sd
    d_arr = dates[day_of]
    cluster = static.cluster.values[sid]
    st_arr = np.array(stations)[sid]

    tr = d_arr <= np.datetime64(TRAIN_END)
    va = (d_arr > np.datetime64(TRAIN_END)) & (d_arr <= np.datetime64(VAL_END))
    te = d_arr > np.datetime64(VAL_END)
    data = dict(Xseq=Xseq, Xctx=Xctx, sid=sid, yz=yz, y=y_raw, mu=mu, sd=sd,
                dates=d_arr, cluster=cluster, station=st_arr,
                masks=dict(train=tr, val=va, test=te), n_stations=n_st)
    print(f"樣本：train {tr.sum():,}｜val {va.sum():,}｜test {te.sum():,}"
          f"｜seq {Xseq.shape}｜ctx {Xctx.shape[1]}維")
    return data


# ---------------------------------------------------------------- 模型
def make_model(n_stations, ctx_dim, hidden, layers, emb_dim, dropout, torch, nn):
    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_stations, emb_dim)
            self.lstm = nn.LSTM(3, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(
                nn.Linear(hidden + ctx_dim + emb_dim, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, 1))

        def forward(self, xseq, xctx, sid):
            _, (h, _) = self.lstm(xseq)
            z = torch.cat([h[-1], xctx, self.emb(sid)], dim=1)
            return self.head(z).squeeze(-1)
    return LSTMNet()


def predict(model, data, mask, device, torch, bs=8192):
    model.eval()
    idx = np.where(mask)[0]
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            p = model(torch.from_numpy(data["Xseq"][j]).to(device),
                      torch.from_numpy(data["Xctx"][j]).to(device),
                      torch.from_numpy(data["sid"][j]).to(device))
            out[i:i + len(j)] = p.cpu().numpy()
    return np.maximum(out * data["sd"][idx] + data["mu"][idx], 0)  # 反z-score並clip


def train_model(data, cfg, fit_mask, es_mask, max_epochs, patience, device, torch, nn, log, tag):
    """fit_mask上訓練；es_mask非None時做early stopping並回傳最佳epoch。"""
    g = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    model = make_model(data["n_stations"], data["Xctx"].shape[1],
                       cfg["hidden"], cfg["layers"], cfg["emb_dim"], cfg["dropout"],
                       torch, nn).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = nn.HuberLoss(delta=1.0)
    fit_idx = np.where(fit_mask)[0]
    bs = cfg["batch"]
    best = dict(wape=np.inf, epoch=0, state=None)
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(fit_idx), generator=g).numpy()
        t0, tot = time.time(), 0.0
        for i in range(0, len(fit_idx), bs):
            j = fit_idx[perm[i:i + bs]]
            opt.zero_grad()
            p = model(torch.from_numpy(data["Xseq"][j]).to(device),
                      torch.from_numpy(data["Xctx"][j]).to(device),
                      torch.from_numpy(data["sid"][j]).to(device))
            loss = lossf(p, torch.from_numpy(data["yz"][j]).to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        row = {"config": tag, "epoch": ep, "train_loss": tot / len(fit_idx),
               "sec": round(time.time() - t0, 1)}
        if es_mask is not None:
            vp = predict(model, data, es_mask, device, torch)
            vw = wape(data["y"][es_mask], vp)
            row["val_WAPE"] = vw
            print(f"  [{tag}] epoch {ep:02d} loss={row['train_loss']:.4f} val WAPE={vw:.3f}% ({row['sec']}s)")
            if vw < best["wape"]:
                best = dict(wape=vw, epoch=ep,
                            state={k: v.cpu().clone() for k, v in model.state_dict().items()})
            elif ep - best["epoch"] >= patience:
                print(f"  [{tag}] early stop（best epoch {best['epoch']}, val WAPE {best['wape']:.3f}%）")
                log.append(row)
                break
        else:
            print(f"  [{tag}] epoch {ep:02d} loss={row['train_loss']:.4f} ({row['sec']}s)")
        log.append(row)
    return model, best


def main(quick=False):
    import torch
    import torch.nn as nn
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"裝置：{device}")

    data = build_dataset()
    m = data["masks"]

    grid = [dict(hidden=64, layers=1, emb_dim=12, dropout=0.1, lr=1e-3, batch=1024),
            dict(hidden=128, layers=1, emb_dim=12, dropout=0.1, lr=1e-3, batch=1024),
            dict(hidden=64, layers=2, emb_dim=12, dropout=0.2, lr=1e-3, batch=1024)]
    max_epochs, patience = 60, 6
    if quick:
        grid, max_epochs, patience = grid[:1], 3, 3

    # 調參：train訓練、val早停
    log, results = [], []
    for k, cfg in enumerate(grid):
        tag = f"cfg{k}_h{cfg['hidden']}L{cfg['layers']}"
        print(f"\n=== 配置 {tag}: {cfg} ===")
        _, best = train_model(data, cfg, m["train"], m["val"],
                              max_epochs, patience, device, torch, nn, log, tag)
        results.append({"tag": tag, **cfg, "best_epoch": best["epoch"], "val_WAPE": best["wape"]})
    res_df = pd.DataFrame(results).sort_values("val_WAPE")
    print("\n=== 配置比較（val WAPE）===")
    print(res_df[["tag", "best_epoch", "val_WAPE"]].round(3).to_string(index=False))
    best_cfg = grid[int(res_df.index[0])]
    best_ep = int(res_df.iloc[0].best_epoch)

    # 最終：train+val重訓固定epoch，test只評一次
    print(f"\n=== 最終重訓（train+val, {best_ep} epochs）===")
    fit_mask = m["train"] | m["val"]
    model, _ = train_model(data, best_cfg, fit_mask, None, best_ep, 0,
                           device, torch, nn, log, "final")
    pd.DataFrame(log).to_csv(REPORTS / "stage7_training_log.csv", index=False)

    p = predict(model, data, m["test"], device, torch)
    te = m["test"]
    rows = [{"model": "LSTM", "group": "ALL", **metrics(data["y"][te], p)}]
    cl_te = data["cluster"][te]
    for cl in sorted(set(cl_te)):
        cm = cl_te == cl
        rows.append({"model": "LSTM", "group": cl, **metrics(data["y"][te][cm], p[cm])})
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "stage7_dl_metrics.csv", index=False)
    pd.DataFrame({"station": data["station"][te], "date": data["dates"][te],
                  "y": data["y"][te], "pred": p}).to_csv(
        REPORTS / "stage7_dl_predictions.csv", index=False)

    print("\n================ LSTM 結果 ================")
    print(res.round(2).to_string(index=False))
    print("\n=== 累積比較表（test WAPE%）===")
    print(cumulative_table(res.iloc[0].WAPE))


def cumulative_table(lstm_wape):
    """從Stage 5/6的實際輸出檔讀取對照數字，不寫死。

    寫死曾造成問題：SARIMAX因Kalman filter狀態初始化修正由6.07%降至4.34%後，
    各腳本內的硬編碼數字全部過期。改為動態讀取可避免同類錯誤重演。
    """
    parts = []
    try:
        b = pd.read_csv(REPORTS / "stage5_baseline_metrics.csv")
        b = b[b.group == "ALL"].set_index("model").WAPE
        for k, lab in [("naive_lag1", "naive"), ("seasonal_naive_lag7", "seasonal naive"),
                       ("rolling28_mean", "rolling28"), ("ols_per_station", "逐站OLS")]:
            if k in b.index:
                parts.append(f"{lab} {b[k]:.2f}")
    except FileNotFoundError:
        pass
    try:
        sp = pd.read_csv(REPORTS / "stage5_sarima_predictions.csv")
        parts.append("SARIMAX %.2f" % (np.abs(sp.y - sp.pred).sum() / sp.y.sum() * 100))
    except FileNotFoundError:
        pass
    try:
        r6 = pd.read_csv(REPORTS / "stage6_ml_metrics.csv")
        r6 = r6[(r6.group == "ALL") & (r6.ablation == "full")]
        for _, r in r6.iterrows():
            parts.append(f"{r.model} {r.WAPE:.2f}")
    except FileNotFoundError:
        pass
    parts.append(f"**LSTM {lstm_wape:.2f}**")
    return " | ".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="單配置3epoch快速驗證環境")
    args = ap.parse_args()
    main(quick=args.quick)
