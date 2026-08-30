# 特徵字典（Feature Dictionary）

> 本檔整理 `data/features/feature_table.csv` 的全部欄位定義、各模型實際使用的變數，以及各特徵的實測貢獻度。
> 資料來源：`scripts/stage4_features.py`（特徵建構）、`scripts/stage5_baseline.py`、`scripts/stage6_ml.py`、`scripts/stage7_dl.py`（各模型特徵集）、`reports/stage8_shap_importance.csv`（SHAP 重要度）。
> 最後更新：2026-08-25

---

## 1. 特徵表總覽

`data/features/feature_table.csv`：**148,393 列**（119 站 × 1,247 天）× **27 欄**。

因 `roll28_mean` 需 28 天回溯期，各站前 28 天為空值，建模時整列剔除（119 × 28 = 3,332 列），**實際訓練與評估使用 145,061 列**，涵蓋 2023-01-29 ～ 2026-05-31。

### 1.1 識別與目標

| 欄位 | 型別 | 說明 |
|---|---|---|
| `station` | 文字 | 站名（119 站，已合併站名前綴變體） |
| `date` | 日期 | 2023-01-01 ～ 2026-05-31 |
| `entries` | 整數 | **預測目標**：該站當日進站量（人次） |

### 1.2 日曆特徵

| 欄位 | 型別 | 定義 | 備註 |
|---|---|---|---|
| `dow` | 0–6 | 星期幾（0 = 週一） | 三個模型各用不同編碼，見 §3 |
| `month` | 1–12 | 月份 | |
| `is_weekend` | 0/1 | `dow >= 5` | **僅看星期，不看是否補班** |
| `is_holiday` | 0/1 | 該日屬**平日**的國定假日 | 名稱易誤導，實為 `is_weekday_holiday`；落在週末的假日此欄為 0 |
| `is_makeup_workday` | 0/1 | 補班日（週末但需上班） | 全期 8 天 |
| `is_offday` | 0/1 | `(is_weekend & ~is_makeup_workday) \| is_holiday` | **實際是否放假**；為前三欄的確定性函數，屬預先編碼的交互特徵 |
| `holiday_name` | 文字 | 假日名稱 | **不參與建模**，僅供人工檢視 |
| `is_lny` | 0/1 | 春節期間 | |
| `is_typhoon` | 0/1 | 颱風停班日 | 全期 6 天 |
| `typhoon_name` | 文字 | 颱風名稱 | **不參與建模**，僅供人工檢視 |
| `is_nye` | 0/1 | 跨年夜（12/31） | |

**四種日型組合的實際分布**（共 1,247 天）：

| `is_weekend` | `is_holiday` | `is_makeup_workday` | `is_offday` | 天數 | 例子 |
|:---:|:---:|:---:|:---:|---:|---|
| 0 | 0 | 0 | 0 | 836 | 一般平日 |
| 1 | 0 | 0 | 1 | 349 | 一般週末 |
| 0 | 1 | 0 | 1 | 54 | 落在平日的國定假日 |
| 1 | 0 | 1 | 0 | **8** | 補班日 |

`is_offday` 存在的唯一理由即最後一列：僅用 `is_weekend | is_holiday` 判斷會把這 8 天誤判為放假。

### 1.3 動量特徵

全部先 `shift(1)` 再計算，不含當日資訊（防洩漏鐵律第一條）。

| 欄位 | 定義 |
|---|---|
| `lag1` | 前 1 日進站量 |
| `lag7` | 前 7 日進站量（上週同一天） |
| `lag14` | 前 14 日進站量 |
| `roll7_mean` | 前 7 日移動平均 |
| `roll28_mean` | 前 28 日移動平均（決定建模起始日 2023-01-29） |

### 1.4 站點靜態特徵

**僅以訓練期（2023-01-01 ～ 2024-12-31）資料計算**（防洩漏鐵律第二條）。

| 欄位 | 型別 | 定義 |
|---|---|---|
| `cluster` | 類別 | 站型標籤：`residential` (46)／`employment` (28)／`employment_peak` (7)／`tourism_hub` (38)。以「平日 21 時段 + 週末 21 時段」共 42 維進站曲線做總量正規化後 k-means (k=4) |
| `pagerank` | 浮點 | OD 有向加權圖上的 PageRank（damping 0.85、冪迭代 200 次） |
| `out_strength_train` | 整數 | 加權出度 = 該站發出的累計旅次（≒ 訓練期累計進站量） |
| `in_strength_train` | 整數 | 加權入度 = 抵達該站的累計旅次 |
| `deg_out` | 整數 | 無權出度 = 目的地站數。**全 119 站恆為 119** |
| `deg_in` | 整數 | 無權入度 = 起點站數。**全 119 站恆為 119** |
| `is_yline` | 0/1 | 是否為環狀線 14 站 |
| `service_suspended` | 0/1 | 0403 地震停駛期間（站 × 日交互旗標） |

> ⚠️ **`deg_out` / `deg_in` 為常數欄位。** 訓練期累計兩年，119 × 119 個站對全部至少有一筆旅次（最冷門的「萬芳社區→Y板橋」僅 1 人次），故 `(A > 0)` 全為真。兩欄對模型零貢獻，SHAP 實測為 0.0（見 §4）。

> ⚠️ **`in_strength_train` 存在制度混合。** 環狀線 14 站的出站紀錄自 2023-06 起消失，故其入流僅累計 5 個月、出流累計 24 個月。純環狀 11 站的 in/out 比值為 0.19–0.36，雙線 3 站為 0.94–0.95，其餘 105 站中位數為 1.005。

---

## 2. 各模型使用的變數對照

| 變數 | 逐站 OLS | SARIMAX | LightGBM | RandomForest | LSTM |
|---|:---:|:---:|:---:|:---:|:---:|
| `entries`（目標） | y | endog | y | y | y（z-score） |
| **日曆** | | | | | |
| `dow` | ✓ 6 dummies | − | ✓ 數值 | ✓ 數值 | ✓ 7 維 one-hot |
| `month` | − | − | ✓ 數值 | ✓ 數值 | ✓ sin/cos 2 維 |
| `is_weekend` | − | − | ✓ | ✓ | − |
| `is_holiday` | − | − | ✓ | ✓ | ✓ |
| `is_offday` | ✓ exog | ✓ exog | ✓ | ✓ | ✓ |
| `is_makeup_workday` | ✓ exog | ✓ exog | ✓ | ✓ | ✓ |
| `is_lny` | ✓ exog | ✓ exog | ✓ | ✓ | ✓ |
| `is_typhoon` | − | − | ✓ | ✓ | ✓ |
| `is_nye` | ✓ exog | ✓ exog | ✓ | ✓ | ✓ |
| **動量** | | | | | |
| `lag1` | ✓ | ARIMA 內生 | ✓ | ✓ | 序列取代 |
| `lag7` | ✓ | ARIMA 內生 | ✓ | ✓ | 序列取代 |
| `lag14` | ✓ | ARIMA 內生 | ✓ | ✓ | 序列取代 |
| `roll7_mean` | ✓ | − | ✓ | ✓ | 序列取代 |
| `roll28_mean` | ✓ | − | ✓ | ✓ | 序列取代 |
| **站點靜態** | | | | | |
| `cluster`（`cl_*` 4 維） | − | − | ✓ | ✓ | ✓ |
| `pagerank` | − | − | ✓ | ✓ | ✓ |
| `out_strength_train` | − | − | ✓ | ✓ | ✓ log1p |
| `in_strength_train` | − | − | ✓ | ✓ | ✓ log1p |
| `deg_out` | − | − | ✓ | ✓ | ✓ |
| `deg_in` | − | − | ✓ | ✓ | ✓ |
| `is_yline` | − | − | ✓ | ✓ | ✓ |
| `service_suspended` | − | − | ✓ | ✓ | ✓ |
| **站點身分** | 一站一模型 | 一站一模型 | 無（靠靜態特徵） | 無 | station embedding 12 維 |
| **特徵總數** | 16 個係數 | 4 exog | 25 欄 | 25 欄 | 序列 3×28 ＋ ctx 26 ＋ emb 12 |

### 各模型的規格細節

**逐站 OLS**（`stage5_baseline.py`，119 個獨立模型）
截距 1 ＋ `dow` dummies 6（`dow=6` 為基準）＋ 4 個事件旗標 ＋ 5 個動量特徵（除以 1e4 縮放）＝ 16 個係數。

**SARIMAX**（`stage5_baseline.py`，119 個獨立模型）
`order=(1,0,1)`、`seasonal_order=(1,1,1,7)`；外生變數僅 4 個：`is_offday`、`is_lny`、`is_nye`、`is_makeup_workday`。動量不作為輸入變數——AR 項與週季節差分本身即在描述時間依賴，這是 ARIMA 家族與其他模型最根本的結構差異。

**LightGBM / RandomForest**（`stage6_ml.py`，各 1 個 global 模型）
`CAL_FEATS`(9) + `LAG_FEATS`(5) + `NET_FEATS`(5) + `OTHER_STATIC`(2) + `cl_cols`(4) = **25 欄**。
兩模型特徵集完全相同，差異純粹來自演算法——這正是兩者幾乎打平（4.63% vs 4.62%）的原因。
消融實驗的 `no_network` arm 移除 `NET_FEATS` 5 欄，**但保留 `cl_cols`**。

**LSTM**（`stage7_dl.py`，1 個 global 模型）
- 序列輸入：28 天 × 3 通道 = `entries` 的 z-score、`is_offday`、`is_holiday`
- 目標日情境：**26 維** = `dow` one-hot(7) + `month` sin/cos(2) + 6 個日級旗標 + `service_suspended`(1) + 靜態數值(5) + `cl_*`(4) + `is_yline`(1)
- station embedding：12 維可學習向量

---

## 3. 編碼差異註記

**同一個 `dow`，三種編碼。** 逐站 OLS 用 6 個 dummy（`dow=6` 為基準，避免共線）；LightGBM／RF 直接當數值（樹可用 `dow >= 5` 這類切分自行處理）；LSTM 用 7 維 one-hot（神經網路需要類別展開）。

**`month` 的 sin/cos 編碼**（僅 LSTM）：月份是循環的，直接餵 1–12 會讓模型誤以為 12 月與 1 月相距 11。改以 $\theta = 2\pi m/12$ 對應到圓上，用 $(\sin\theta, \cos\theta)$ 表示，相鄰月份即等距，且 12 月與 1 月確實相鄰。需要兩維才能唯一確定圓上位置。

**LSTM 沒有 `is_weekend`**：`dow` 已展開成 7 維 one-hot，`is_weekend` 等於其中第 5、6 維相加，對神經網路完全冗餘（第一層線性權重可自行合成）。樹模型無法自動做這種線性組合，故保留——但 SHAP 顯示它實際上也沒被用到（見 §4）。

**LSTM 沒有動量欄位**：直接讀 28 天原始序列，人工 lag/rolling 特徵是多餘的。

**逐站模型沒有站點靜態特徵**：一站一模型時，站的身分已由「這是第幾個模型」表達，`pagerank` 這類站內恆定的值毫無資訊量。這也是消融實驗只能在 global 模型上進行的原因。

---

## 4. 實測貢獻度（LightGBM 全域 SHAP）

來源：`reports/stage8_shap_importance.csv`（`group = ALL`，mean |SHAP| 佔比）

| 排名 | 特徵 | 佔比 % | 分類 |
|---:|---|---:|---|
| 1 | `lag7` | 31.66 | 動量 |
| 2 | `lag14` | 15.93 | 動量 |
| 3 | `pagerank` | 12.54 | 網絡 |
| 4 | `lag1` | 12.42 | 動量 |
| 5 | `is_offday` | 8.42 | 日曆 |
| 6 | `out_strength_train` | 6.07 | 網絡 |
| 7 | `roll7_mean` | 4.03 | 動量 |
| 8 | `is_holiday` | 2.20 | 日曆 |
| 9 | `dow` | 2.16 | 日曆 |
| 10 | `month` | 1.20 | 日曆 |
| 11 | `in_strength_train` | 1.11 | 網絡 |
| 12 | `roll28_mean` | 1.04 | 動量 |
| 13 | `is_typhoon` | 0.54 | 稀有事件 |
| 14 | `cl_employment_peak` | 0.19 | 站型 |
| 15 | `cl_tourism_hub` | 0.12 | 站型 |
| 16 | `is_lny` | 0.11 | 稀有事件 |
| 17 | `cl_employment` | 0.10 | 站型 |
| 18 | `is_nye` | 0.07 | 稀有事件 |
| 19 | `cl_residential` | 0.04 | 站型 |
| 20 | `is_yline` | 0.02 | 靜態 |
| 21 | `service_suspended` | 0.013 | 稀有事件 |
| 22 | `is_makeup_workday` | 0.006 | 日曆 |
| 23 | `is_weekend` | **0.00** | 日曆 |
| 24 | `deg_out` | **0.00** | 網絡 |
| 25 | `deg_in` | **0.00** | 網絡 |

### 4.1 三個完全未被使用的特徵

`is_weekend`、`deg_out`、`deg_in` 的 mean |SHAP| **精確為 0.0**——LightGBM 從未對它們建立任何切分。

- `deg_out`／`deg_in`：常數欄位（恆為 119），樹模型無法產生 gain > 0 的切分。
- `is_weekend`：`is_offday` 與 `dow` 已完整涵蓋其資訊，模型找不到使用它的理由。

換言之，25 欄特徵中**實際發揮作用的只有 22 欄**。消融實驗移除的 5 個網絡特徵中，`deg_out`／`deg_in` 兩欄本來就無作用，該次 0.362pp 的退步全部來自 `pagerank` 與兩個 `strength`。

### 4.2 稀有事件旗標幾乎沒有話語權

`is_typhoon` 0.54%、`is_lny` 0.11%、`is_nye` 0.07%、`service_suspended` 0.013%，**四者合計僅 0.74%**。

原因是這些日子在訓練期太稀有，訊號被其餘 99% 以上的一般日子稀釋掉——一套參數要同時擬合 119 站 × 上千天，優化方向必然偏向「對多數樣本最好」，稀有事件撐不起一條專屬的判斷路徑。

**這是第 8 節所有現象的共同根源**：極端節日的世代反轉、颱風 hold-out 的 zero-shot 失效、事件型場站的高誤差，都可以回溯到這 0.74%。

### 4.3 動量與網絡的分工

動量特徵合計 **65.1%**（`lag7` + `lag14` + `lag1` + `roll7_mean` + `roll28_mean`），其中週節律相關的 `lag7` + `lag14` 即佔 47.6%——與「seasonal naive 大幅優於 naive」互相印證。

網絡特徵合計 **19.7%**（`pagerank` + 兩個 `strength`），扮演的是 global 模型辨識站點量體的角色。站型 one-hot 合計僅 0.45%，遠低於網絡特徵——顯示模型偏好用連續的量體刻度而非四分類標籤來區分站點。

### 4.4 站型間的差異指紋

| 特徵 | 全網 | 住宅型 | 就業型 | 就業極端型 | 轉運觀光型 |
|---|---:|---:|---:|---:|---:|
| `lag7` | 31.66 | **34.54** | 33.77 | 33.83 | 27.09 |
| `pagerank` | 12.54 | 11.14 | 5.51 | 9.83 | **19.88** |
| 網絡特徵合計 | 19.72 | 18.84 | 12.98 | 16.93 | **26.41** |
| `is_offday` | 8.42 | 9.12 | 8.76 | 8.84 | 7.47 |

住宅型最依賴週節律（`lag7` 34.5%，全場最高），故逐站 SARIMAX 已足夠；轉運觀光型最依賴網絡位置（`pagerank` 19.9%、網絡合計 26.4%，皆為全場最高）且週節律最弱（27.1%）——規律性差，模型只能改以靜態身分錨定水準。

> **一個容易誤讀的地方：SHAP 佔比 ≠ 效能貢獻。** 轉運觀光型的網絡特徵佔比最高（26.4%），但消融實驗顯示移除網絡特徵對它的損害（0.410pp）反而**低於**就業極端型（0.663pp）。SHAP 衡量「模型輸出隨該特徵變動的幅度」，消融衡量「移除後無可替代的資訊損失」——**重要度高不等於不可取代**。

---

## 5. 三條防洩漏規則對應到的欄位

1. **動量特徵一律先 `shift(1)` 再計算**：`lag1`、`lag7`、`lag14`、`roll7_mean`、`roll28_mean` 皆不含當日資訊。
2. **站點靜態特徵只用訓練期計算**：`cluster`、`pagerank`、`out_strength_train`、`in_strength_train`、`deg_out`、`deg_in` 均以 `date <= 2024-12-31` 的資料建構，即使全期計算「比較準」也拒絕。
3. **事件旗標採事前可得的官方公告**：`is_typhoon` 來自前一晚人事行政總處公布，`is_holiday`／`is_makeup_workday` 來自政府行政機關辦公日曆表，絕不由當日運量反推。
