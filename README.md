# 台北捷運人流的智慧化預測
### 從古典統計到機器學習與深度學習的比較研究，並整合路網流動與事件效應

以 **3.41 億筆**臺北捷運進出站 OD 紀錄（2023-01～2026-05，119 站 × 1,247 天），比較三世代的預測方法——SARIMAX、LightGBM/RandomForest、LSTM——預測「每站次日進站量」，並回答一個比模型排名更實用的問題：

> **不同「個性」的車站（住宅／就業／就業極端／轉運觀光），適合的預測工具一樣嗎？**

答案是**不一樣**——而且最複雜的模型並非處處最好。

📄 完整書面報告見 **`MRT_Project_V1.pdf`**；零基礎讀者建議先讀 **`PROJECT_GUIDE.md`**。

## 主要結果（test = 2026/01–05、一步預測、WAPE，越低越好）

| 模型 | 世代 | 整體 | 住宅型 | 就業型 | 就業極端型 | 觀光型 |
|---|---|---|---|---|---|---|
| rolling28（抄月均） | 基準 | 15.83% | 15.36% | 16.34% | 35.80% | 13.63% |
| naive（抄昨天） | 基準 | 13.44% | 13.45% | 13.75% | 25.07% | 11.95% |
| seasonal naive（抄上週） | 基準 | 9.88% | 8.70% | 10.45% | 13.23% | 9.84% |
| 逐站 OLS ×119 | 統計 | 4.83% | 4.17% | 4.95% | 5.34% | 5.15% |
| LightGBM global | ML | 4.63% | 3.93% | **4.62%** | 4.26% | 5.20% |
| RandomForest global | ML | 4.62% | 3.72% | 4.71% | 4.45% | 5.21% |
| SARIMAX 逐站 ×119 | 統計 | 4.34% | **3.23%** | **4.62%** | **3.17%** | 5.02% |
| **LSTM global** | DL | **3.99%** | 3.29% | **3.97%** | 3.60% | **4.55%** |

**核心發現**：

1. **沒有任何模型處處最好**——LSTM 整體最佳（3.99%），但只在 **54/119 站**優於逐站 SARIMAX，且在住宅型與就業極端型輸給它。量體與勝負確有相關（最大四分位勝率 25/30、最小四分位僅 8/30），這是全域模型的損失涵蓋所有站、容量向貢獻多數運量的大站靠攏，小站則是逐站模型主場的**結構性分工**。但就業極端型 0/7 是另一個獨立因素：這 7 站量體橫跨 Q1~Q4（日均進站約 3.3 萬、全站第 11 大的松江南京也照樣輸），真正驅動的是「單一晚峰、近乎決定性」的**模式規律性**，不是量體。
2. **「更複雜 = 更準」不成立**：修正後的 SARIMAX（4.34%）勝過兩個樹模型（4.62–4.63%）。決定成績的是**模型與資料生成結構的匹配度**，不是模型容量——就業極端型的通勤節律極規則，週季節差分 `(1,0,1)(1,1,1)₇` 正好對症（3.17%，全場最佳）。
3. **站型決定工具的邊際價值**：住宅型與就業極端型用逐站 SARIMAX 即可（連 LSTM 都贏不了）；觀光型在統計與 ML 世代全部卡在 5.02–5.21%，唯有 LSTM 突破至 4.55%。就業極端型的跨世代跨度最大——從 rolling28 的 35.80% 到 SARIMAX 的 3.17%，**相差 11.3 倍**，難度完全由「方法能否表達星期節律」決定。
4. **路網特徵有增量價值**（消融實驗）：拿掉 PageRank／流量強度／度數，WAPE 4.634%→4.996%（+0.362pp），其中就業極端型受傷最重（+0.663pp，為住宅型的 3.5 倍）。SHAP 顯示 PageRank 依 gain 排第 1（佔 45.31%）。
5. **極端節日出現世代反轉**：春節與元旦皆由逐站 SARIMAX 勝出（15.35%／12.39%，對照 LightGBM 20.52%／28.27%）——逐站模型的節日係數各自估計，而全域模型的稀有旗標因訓練樣本過少難以穩定學習（春節、跨年、颱風三個旗標的 SHAP 佔比合計僅 0.73%）。
6. **事件旗標有效，但模型必須「經歷過」**（hold-out 實驗，訓練嚴格截至事件前）：訓練期未出現過該旗標值時，樹模型無法建立切分，有無旗標預測**完全相同**；經歷過一次以上即可消除七至八成誤差。無法編碼的事件（新北耶誕城開幕）造成連續三天約 7% 的系統性低估。事件旗標的 SHAP 佔比全部 <1%——觀光站缺的確實是資訊，不是模型。

> **⚠️ 方法論註記（2026-08）**：本專案在複查中發現並修正了兩個影響結論的實作缺陷，兩者都已寫入報告作為方法論案例：
> 1. **SARIMAX 的 Kalman filter 狀態初始化**：測試期 filtering 僅提供 14 天暖身，觸發 diffuse initialization，使 WAPE 被高估 **1.73pp**（6.07% → 4.34%）。修正後「ML 世代勝出」的結論反轉。
> 2. **特徵重要度基準**：`LGBMRegressor.feature_importances_` 預設回傳 split 而非 gain，兩者給出完全不同的排序（PageRank 依 split 排第 8、依 gain 排第 1）。
>
> **評估基準線時，實作正確性的風險與模型選擇的風險同量級。**

## 專案亮點

- **大型資料工程**：12.4GB 原始 CSV → DuckDB + parquet(ZSTD) 207MB，一台筆電可分析 3.41 億筆。
- **嚴格防洩漏**：lag/rolling 一律 shift(1)；站點靜態特徵（k-means 分群、PageRank）僅用訓練期計算；事件旗標採「事前可得」官方公告。
- **公平比較協定**：所有模型共用同一切分（train 2023-01-29～2024-12-31 / val 2025 全年 / test 2026-01-01～05-31）、同一指標（WAPE）、test 只評一次。
- **分群數經過驗證**：k=4 以 elbow／silhouette／gap 三指標診斷（`data/agg/cluster_k_diagnostics.csv`），並以下游消融實驗（`reports/stage6b_k_ablation_matrix.csv`、`stage6b_k_ablation_boot.csv`）確認 k=3 與 k=4 對預測效能無顯著差異——分群的功能是分層評估與解釋，不是提升精度。
- **OD 路網分析**：以進出站對建網絡，PageRank／流量強度作為特徵並做消融驗證。

## 專案結構

```
├── README.md / requirements.txt
├── MRT_Project_V1.pdf            ← 完整書面報告（作品集本體，GitHub 可線上預覽）
├── PROJECT_GUIDE.md              ← 零基礎讀者的完整入門指南（推薦先讀）
├── feature_dictionary_new.md     ← 特徵字典：27 欄的定義、各模型使用的變數、實測貢獻度
├── scripts/                      ← stage1b~8 全部可重現腳本
├── data/
│   ├── parquet/od/               ← 41個月OD資料（207MB，不入版控，stage1b重新下載）
│   ├── agg/                      ← EDA聚合表與診斷（入版控）
│   └── features/                 ← calendar_daily.csv、station_static.csv
│                                    （feature_table.csv 24MB 不入版控，stage4重產）
└── reports/
    ├── *.csv                     ← 各階段 metrics、逐站/逐日預測、tuning log、SHAP 佔比
    └── figures/                  ← 報告用圖 1~9 及附錄圖
```

## 重現全部結果

```bash
pip install -r requirements.txt                      # Python 3.10+
python scripts/stage1b_download_od.py                # 下載41個月OD資料（數小時，可中斷續跑）
python scripts/stage2_eda.py                         # EDA：聚合表 + 圖1~5（約 3-8 分鐘）
python scripts/stage2b_ring_handover_diagnostic.py   # 環狀線移交診斷（報告 §2.2 表1）
python scripts/stage2c_od_sparsity.py                # OD 零流量診斷（報告 §3.3，約 5 分鐘）
python scripts/stage4_features.py                    # 特徵表 feature_table.csv
python scripts/stage5_baseline.py                    # 基準線 + SARIMAX（約 10-30 分鐘）
python scripts/stage6_ml.py                          # RF + LightGBM + 網絡特徵消融
python scripts/stage6b_k_ablation.py                 # 分群粒度消融（k=3 vs k=4）
python scripts/stage7_dl.py                          # LSTM（CPU 約 15 分鐘）
python scripts/stage8_shap_events.py                 # SHAP + 事件 case study（約 5-8 分鐘）
```

順序不可顛倒（後面腳本依賴前面產出）。所有隨機過程固定 seed=42。

日曆特徵（假日／補班／春節／跨年）與颱風停班日清單已直接寫入 `scripts/stage4_features.py`，
無須另外下載，因此**唯一的外部依賴是 `stage1b` 抓取的北捷 OD 開放資料**。

`scripts/stage0_data_audit.py` 與 `stage1_data_pipeline.py` 是早期針對手動下載 CSV 的
盤點／轉檔工具，功能已被 `stage1b` 涵蓋，不在重現流程中。

> **關於圖 1**：報告中的圖 1 為手動調整標籤位置的版本（`reports/figures/fig1.png`）；
> `stage2_eda.py` 的程式產出為 `fig1_daily_ridership.png`，資料完全相同，
> 差別僅在事件標籤未做防重疊處理。

## 資料來源

- 臺北捷運各站分時進出量統計 OD（臺北市資料大平臺，`stage1b` 自動下載）
- 政府行政機關辦公日曆表（data.gov.tw，dataset 14718）——已整理為 `stage4_features.py` 內的日曆常數
- 臺北市歷次天然災害停止上班上課訊息（臺北市資料大平臺）——已整理為 `stage4_features.py` 內的颱風清單

## 限制與 Future Work

- 事件特徵僅到「全市級」（假日／颱風／跨年），場站級事件（演唱會、展覽）為主要誤差來源與改進方向。
- 樹模型對訓練期未見過的旗標值無法外推（0403 地震停駛為例）。
- LSTM 的序列通道僅含進站量、實際放假日與國定假日，停駛旗標只描述目標日，模型無法從序列得知過去 28 天是否停駛。
- 測試期無颱風日，颱風泛化能力以 hold-out 實驗補充，而非測試期直接觀察。

---
*作者：Sherry（liushuyu1219@gmail.com）*
