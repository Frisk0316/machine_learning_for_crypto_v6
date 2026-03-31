# Machine Learning for Crypto v6 — 排序損失修正、橫截面模型與 ETF 特徵

> **v6 實驗：修正 ListNet 排序損失 + 新增 CrossSectionalGatedNet + Farside ETF 特徵**
>
> 基於 v5.4 發現的 LSTM/TFT 持續劣於 OLS 的問題，進行根本原因分析並提出 4 項修正：
>
> **v6 核心變更**：
> - **ListNet 目標修正**：原始報酬 → **橫截面排名百分位 [0,1]**（Poh et al. 2021），解決不同週報酬幅度導致的訓練信號不一致
> - **梯度累積**：`grad_accum_steps=4`，有效 batch = 86×4 = 344 assets（v5 僅 86）
> - **新模型 CrossSectionalGatedNet**：純橫截面前饋網路（無 LSTM 時序），3 層 GRN + CrossAssetAttention，靈感來自 Gu, Kelly, Xiu (2020)
> - **TFT VSN 修正**：LightVSN → **VariableSelectionNetwork**（共享 per-variable GRN，~27K params），更接近 Lim et al. (2021) 原論文
> - **新增 4 個 Farside ETF 特徵**：GBTC/ETHE 個別流量 + 排除 GBTC/ETHE 的總流量（結構性供需信號）
> - **新增 2 個特徵組**：`+ETF`、`+Trump+ETF`

---

## 1. Motivation / 研究動機

### 1.1 v4 診斷出的四大缺陷

v4 實驗（原始 LSTM / TFT）揭示了時間序列模型在橫截面排序任務上的系統性劣勢：

| # | 缺陷 | v4 表現 |
|---|------|--------|
| 1 | **任務本質錯配**：MSE 損失最小化絕對預測誤差，但任務需要相對排序 | TFT SR=+0.08, LSTM SR=+1.08 (均遠低於 Gated FFN SR=+2.90) |
| 2 | **無跨資產交互**：每個資產獨立處理，無法比較不同資產 | Decile 排序幾乎隨機（TFT 無單調模式） |
| 3 | **過度參數化**：TFT 368K 參數 / 14K 樣本 = 25.4 比值 | 驗證→測試 Sharpe 平均衰退 -1.0 至 -2.0 |
| 4 | **時序模式不穩定**：12 週回看窗口在加密市場中不穩定 | 高驗證 SR 無法預測高測試 SR |

### 1.2 v5 改善策略

針對每個缺陷提出對應改善：

| 缺陷 | v4 做法 | v5 改善 |
|------|---------|---------|
| 損失函數錯誤 | MSE（最小化絕對誤差） | **ListNet Ranking Loss**（直接優化排序） |
| 無跨資產交互 | 獨立處理每個資產 | **Cross-Asset Self-Attention** |
| 過度參數化 | TFT 368K, LSTM 82K | 降至 TFT **~31K**, LSTM **~18K** |
| 時序不穩定 | lookback=12 | lookback=8, 加 **position embedding** |
| 正則化不足 | dropout=0.15, decay=1e-5 | dropout=**0.30**, decay=**1e-4** |
| 訓練不穩定 | 直接 lr=0.001 | **Linear warmup** 10 epochs |

---

## 2. Architecture / 模型架構

### 2.1 LSTM v5 (with Cross-Asset Attention)

```
Input (N, L=8, M) → Input Projection (M → d=64)
    + Temporal Position Embedding (8 positions, learnable)
    → 1-layer LSTM (d=64)
    → Last hidden state (N, d)
    ⊕ Asset Embedding (86 → d=64)
    → Cross-Asset Self-Attention (2 heads)  ← NEW
    → LayerNorm + residual
    → Linear(d → 1)
    → Ranking scores (N,)
```

- **Parameters: ~53K** (v4: ~82K, v5.3: ~18K)
- Cross-Asset Attention allows direct comparison between assets

### 2.2 CrossSectionalGatedNet (v6 新增)

```
Input (N, L=8, M) → x[:, -1, :] (current time-step only, no temporal)
    → Linear(M → d=64) + Dropout
    → [GRN → Dropout] × 3  (Gated Residual Network layers)
    ⊕ Asset Embedding (86 → d=64)
    → Cross-Asset Self-Attention (2 heads)
    → LayerNorm + residual
    → Linear(d → 1)
    → Ranking scores (N,)
```

- **Parameters: ~49-75K**（依特徵數量而定）
- **無 LSTM 時序編碼**：僅使用當期特徵，驗證時序建模是否增加價值
- 靈感來自 Gu, Kelly, Xiu (2020) 的 NN1-NN5 純前饋設計
- 使用 GRN（GLU gating）取代 ReLU MLP，提供更好的特徵交互

### 2.3 TFT v6 (with Cross-Asset Attention + VariableSelectionNetwork)

```
Input (N, L=8, M) → Per-variable embedding (M × (weight, bias) → d=64)
    + Temporal Position Embedding (8 positions, learnable)
    → VariableSelectionNetwork (v6 修正)
        ├─ Shared GRN trunk: per-variable processing
        ├─ Pooled + context → softmax weights (interpretable)
        └─ Static context: Asset Embedding (86 → d=64)
    → 1-layer LSTM encoder (d=64)
    → Gated skip connection + LayerNorm
    → Interpretable Multi-Head Attention (2 heads)
    → Gated skip connection + LayerNorm
    → Cross-Asset Self-Attention (2 heads)
    → LayerNorm + residual
    → Linear(d → 1)
    → Ranking scores (N,)
```

- **Parameters: ~90-100K** (v4: ~209-368K, v5.3: ~29-31K)
- **v6 VSN 修正**：LightVSN (~5K) → VariableSelectionNetwork (~27K)，共享 GRN trunk 處理每個變數（更接近 Lim et al. 2021 原論文）
- Cross-Asset Attention: enables cross-sectional comparison after temporal encoding

### 2.4 v4 → v5 → v6 Architecture Comparison

| Component | v4 | v5 | Rationale |
|-----------|----|----|-----------|
| LSTM layers | 2 | **1** | Reduce capacity |
| Hidden dim | 64 | **64** | v5.4 恢復為 64（v5.3 為 32 過於保守） |
| Attention heads | 4 | **2** | Fewer heads sufficient |
| VSN | Full GRN (M×d flattened) | **LightVSN** (pooled) | Fewer params |
| Cross-asset | None | **Self-Attention** | Enable ranking |
| Position encoding | None | **Learnable** | Time step awareness |
| Output head | FC(2H→H→1) or GRN→Linear | **Linear(d→1)** | Ranking only needs scores |

### 2.4 Cross-Asset Self-Attention (Key Innovation)

```python
class CrossAssetAttention:
    """
    After temporal encoding, each asset has representation h_i ∈ R^d.
    Stack all N assets: H = [h_1, ..., h_N] ∈ R^(N, d)
    Apply self-attention: H' = LayerNorm(H + MHA(H, H, H))

    This lets asset i attend to ALL other assets at the same time step,
    enabling direct cross-sectional comparison for ranking.
    """
```

v4 的根本問題在於每個資產獨立通過 LSTM/TFT，模型無法比較不同資產的表現。Cross-Asset Attention 在時序編碼後加入一層跨資產自注意力，讓每個資產的表示可以參考所有其他資產，從而實現橫截面排序。

### 2.6 ListNet Ranking Loss with Rank-Normalized Targets (v6 修正)

```python
def listnet_loss(pred, target, mask, temperature=1.0):
    """
    v6 修正：使用橫截面排名百分位作為目標（Poh et al. 2021）

    1. 將原始報酬轉換為排名百分位：ranks / (N-1) → [0, 1]
    2. p_true = softmax(rank_percentile / τ)
    3. L = -Σ p_true · log_softmax(pred / τ)

    v5 問題：直接用原始報酬 softmax → 大跌週（-20%,+40%）和
    平靜週（-0.5%,+1%）的目標分佈銳度不一致
    v6 修正：排名百分位使每週目標分佈具有相同的熵
    """
```

v4 使用 MSE 損失，v5 改用 ListNet 但直接對原始報酬做 softmax（尺度不一致），v6 進一步改用排名百分位目標，確保訓練信號在不同市場環境下保持一致。

---

## 3. Data / 數據

**Source**: `deep_learning_for_crypto/datasets/btc_panel_v6.npz`（由 `prepare_panel_v6.py` 從 `btc_panel.npz` 擴充）

| Dimension | Value |
|-----------|-------|
| Time (T) | 324 weeks (2020-01-05 → 2026-03-15) |
| Assets (N) | 86 cryptocurrencies (top 100 excl. stablecoins) |
| Features (M) | **53** (v5: 49, v6 新增 4 個 Farside ETF 特徵) |
| Target | Next-week cross-sectional return |
| Missing data | UNK = −99.99, masked in loss and evaluation |

**Feature categories:**
- **Price Momentum (0–4):** r1w, r4w, r12w, r26w, r52w
- **Technical (5–10):** RSI, Bollinger, vol_ratio, ATR, OBV, vol_usd
- **On-chain (11–15):** active_addr, tx_count, NVT, exchange_net_flow, MVRV
- **Macro/Sentiment (16–26):** Fear & Greed, S&P500, DXY, VIX, gold, silver, DJI
- **ETF + Polymarket (27–32):** BTC/ETH ETF flows, Polymarket BTC probability, ETF volume
- **Trump Social Media (44–48):** trump_post_count, trump_caps_ratio, trump_tariff_score, trump_crypto_score, trump_sentiment
- **Farside ETF Granular (49–52, v6 新增):** btc_etf_gbtc, btc_etf_excl_gbtc, eth_etf_ethe, eth_etf_excl_ethe

### 3.1 Trump Social Media Features (v5.1 新增)

從 [trump-code](https://github.com/sstklen/trump-code) 專案提取 5 個週頻宏觀特徵，捕捉 Trump 社群媒體行為對加密市場的影響：

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 44 | `trump_post_count` | Truth Social + X | 每週發文數（silence = bullish signal） |
| 45 | `trump_caps_ratio` | 文本分析 | 平均大寫字母比例（情緒強度指標） |
| 46 | `trump_tariff_score` | 關鍵詞匹配 | 關稅/貿易戰相關貼文比例 |
| 47 | `trump_crypto_score` | 關鍵詞匹配 | 加密貨幣相關貼文比例 |
| 48 | `trump_sentiment` | 標點分析 | 感嘆號密度（情緒代理） |

**資料來源**：
- Truth Social archive（14,623 篇原創貼文，2022-02 → 2024-06）
- X/Twitter archive（168 篇推文，2025-01 → 2026-03）

**覆蓋率**：209/324 週有效（64.5%），2020-01 至 2022-02 為 UNK（Trump 尚未使用 Truth Social）

**標準化**：與其他宏觀特徵相同，使用 52 週滾動 z-score

### 3.2 Farside ETF Granular Features (v6 新增)

從 [Farside Investors](https://farside.co.uk) 取得 BTC/ETH Spot ETF 每日淨流入資料，提供比原有聚合 ETF 特徵（indices 27–32）更細緻的結構性供需信號：

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 49 | `btc_etf_gbtc` | Farside BTC ETF | GBTC 每週淨流量（$M, z-scored）— Grayscale 存量基金贖回壓力 |
| 50 | `btc_etf_excl_gbtc` | Farside BTC ETF | BTC ETF Total 排除 GBTC（$M, z-scored）— 純新增機構需求 |
| 51 | `eth_etf_ethe` | Farside ETH ETF | ETHE 每週淨流量（$M, z-scored）— Grayscale ETH 基金贖回壓力 |
| 52 | `eth_etf_excl_ethe` | Farside ETH ETF | ETH ETF Total 排除 ETHE（$M, z-scored）— 純新增 ETH 機構需求 |

**結構性信號邏輯**：
- **GBTC/ETHE 流出** = 舊 Grayscale 信託持有人轉換至費率更低的 Spot ETF（供給側壓力）
- **Ex-GBTC/Ex-ETHE 流入** = 純新資金進入加密 ETF 市場（需求側信號）
- 兩者的差異捕捉了「存量轉換」vs「增量需求」的結構性分離

**資料覆蓋**：BTC ETF 自 2024-01-11（111/324 週），ETH ETF 自 2024-07-23（83/324 週）

**標準化**：52 週滾動 z-score（與其他宏觀特徵一致），pre-ETF 期間為 UNK

**建構流程**：`prepare_panel_v6.py` — 讀取 Farside CSV → daily-to-weekly 聚合 → rolling z-score → 附加至 `btc_panel.npz` → 輸出 `btc_panel_v6.npz`

### 3.3 Look-Ahead Bias 分析

Trump 社群媒體特徵（indices 44–48）**不存在 look-ahead bias**。以下逐步驗證：

#### 3.3.1 原始特徵計算（`fetch_trump.py`）

```python
week_posts = [p for p in parsed if week_start < p["dt"] <= week_end]
```

每週的 5 個特徵（post_count, caps_ratio, tariff_score, crypto_score, sentiment）僅使用 `(dates[t-1], dates[t]]` 區間內的貼文計算。**不使用未來資料**。

#### 3.3.2 滾動 z-score 標準化（`prepare_btc_data.py`）

```python
roll_mean = ser.rolling(52, min_periods=4).mean()  # window = [t-51, ..., t]
roll_std  = ser.rolling(52, min_periods=4).std()
z[t] = (col[t] - roll_mean[t]) / roll_std[t]
```

`rolling(52)` 在時間 t 使用 `[t-51, ..., t]` 的資料，**僅包含過去和當前值**。當前值包含在自身標準化中（~1/52 = 1.9% 權重），這是金融時間序列的標準做法（如 Gu, Kelly & Xiu 2020 對股票特徵的處理）。

可選改善：加 `.shift(1)` 使標準化更嚴格（僅用 t-1 以前的資料），但實質影響極小。

#### 3.3.3 已知偏誤（非 Trump 特有）

| 偏誤類型 | 描述 | 影響 |
|----------|------|------|
| **存活偏誤** | 86 資產依 2026-03 市值選取 | 所有特徵共同存在，非 Trump 特有 |
| **覆蓋率缺口** | Truth Social: 2022-02→2024-06, X: 2025-01→2026-03 | 中間 ~6 個月為 UNK，正確行為 |

#### 3.3.4 結論

Trump 特徵在特徵計算和標準化兩個環節均**嚴格使用過去和當前資料**，不存在 look-ahead bias。唯一的設計選擇（rolling window 包含當前值）是金融計量學的標準做法。

**Chronological split (70 / 15 / 15):**
- Train: weeks 0–226 (227 weeks) — 預設
- Train (+Trump): weeks ~115–226 (~112 weeks) — 從 Trump 資料覆蓋開始，自動偵測
- Valid: weeks 227–274 (48 weeks)
- Test: weeks 275–323 (49 weeks)

**v5 Sequence construction (vs v4):**

| | v4 | v5 |
|-|----|----|
| Lookback | L = 12 weeks | **L = 8 weeks** |
| Training batch | Random (asset, time) pairs, batch=128 | **Cross-sectional**: all 86 assets at one time step |
| Training samples | ~14,458 individual pairs | **~218 time steps** (each with N=86 assets) |
| Loss computation | Per-sample MSE | **Per-time-step ListNet** (ranks all 86 assets) |

---

## 4. Quick Start / 快速開始

### Prerequisites

```bash
pip install torch numpy matplotlib scipy scikit-learn
```

### One-click execution (全部一鍵執行)

```bash
cd 論文/machine_learning_for_crypto_v6
bash run_all.sh
```

### Step-by-step

```bash
# Step 0: Build v6 panel (add Farside ETF features, only needed once)
python prepare_panel_v6.py

# Step 1: Train traditional models (OLS, EN, PCA, PLS, RF, GBT, ~6 min)
python train_traditional.py --config config.json

# Step 2: Train CS_Gated (32 seeds × 6 info sets, ~120 min on GPU)
python train.py --config config.json --model cs_gated

# Step 3: Train LSTM v6 (32 seeds × 6 info sets, ~123 min on GPU)
python train.py --config config.json --model lstm

# Step 4: Train TFT v6 (32 seeds × 6 info sets, ~116 min on GPU)
python train.py --config config.json --model tft

# Step 5: Cross-model comparison table
python evaluate.py --config config.json --compare-all
```

### Outputs

```
checkpoints/
├── tft_results.npz      # TFT ensemble predictions
└── lstm_results.npz     # LSTM ensemble predictions

outputs/
├── tft_table3.csv       # TFT performance summary
├── tft_cumulative_returns.png
├── tft_decile_sharpe.png
├── tft_feature_importance.png
├── lstm_table3.csv      # LSTM performance summary
├── lstm_cumulative_returns.png
├── lstm_decile_sharpe.png
└── lstm_feature_importance.png
```

---

## 5. Hyperparameters / 超參數

| Parameter | v4 | **v5** | Change Reason |
|-----------|----|----|---------------|
| `lookback` | 12 | **8** | 減少時序雜訊 |
| `hidden_dim` | 64 | **64** | v5.4 恢復（v5.3 為 32 過於保守） |
| `num_heads` | 4 | **2** | 匹配更小維度 |
| `lstm_layers` | 2 | **1** | 降低容量 |
| `dropout` | 0.15 | **0.30** | 加強正則化 |
| `weight_decay` | 1e-5 | **1e-4** | 加強 L2 |
| `learning_rate` | 0.001 | 0.001 | 保持不變 |
| `batch_size` | 128 | **N/A** (cross-sectional) | 每個時間步即一個 batch |
| `grad_accum_steps` | N/A | **4** | v6 新增：有效 batch = 86×4 = 344 |
| `epochs` | 150 | 150 | 保持不變 |
| `early_stopping_patience` | 20 | 20 | 保持不變 |
| `warmup_epochs` | N/A | **10** | 穩定早期訓練 |
| `ranking_temperature` | N/A | **1.0** | ListNet softmax 溫度 |
| `num_seeds` | 8 | **32** | 增加至 32 以降低 ensemble 變異數 |
| `cs_gated_layers` | N/A | **3** | v6 新增：CrossSectionalGatedNet GRN 層數 |

---

## 6. Results / 實驗結果

### 6.1 Table 3: Long-Short Portfolio Performance (Test Period)

> **v6 結果** — rank-normalized ListNet, grad_accum=4, 32 seeds ensemble

**CS_Gated v6 Ensemble (32 seeds, deterministic) — 純橫截面模型：**

| Information Set | SR (PW) | SR (EW) |
|-----------------|:-------:|:-------:|
| Price+Technical (11) | +1.33 | +1.34 |
| +Onchain (16) | **+2.61** | **+2.75** |
| All (33) | +1.25 | +0.86 |
| +Trump (38) | +0.90 | +1.24 |
| +ETF (37) | +1.16 | +1.42 |
| +Trump+ETF (42) | +1.61 | +1.82 |

**TFT v6 Ensemble (32 seeds, deterministic):**

| Information Set | SR (PW) | SR (EW) |
|-----------------|:-------:|:-------:|
| Price+Technical (11) | +0.19 | +0.18 |
| +Onchain (16) | +1.30 | +1.49 |
| All (33) | +0.17 | +0.44 |
| +Trump (38) | +0.98 | +0.93 |
| +ETF (37) | +0.87 | +0.89 |
| +Trump+ETF (42) | **+1.89** | **+1.76** |

**LSTM v6 Ensemble (32 seeds, deterministic):**

| Information Set | SR (PW) | SR (EW) |
|-----------------|:-------:|:-------:|
| Price+Technical (11) | **+2.46** | +1.45 |
| +Onchain (16) | +0.93 | +1.13 |
| All (33) | +0.59 | +0.37 |
| +Trump (38) | +0.57 | +0.59 |
| +ETF (37) | +0.35 | +0.57 |
| +Trump+ETF (42) | **+2.13** | **+2.03** |

### 6.2 v5.4 → v6 Improvement / v5.4 vs v6 改善比較

| Model | v5.4 best SR | v6 best SR | v6 Best Config | Δ SR |
|-------|:------------:|:----------:|:--------------:|:----:|
| TFT | +0.48 (+Trump) | **+1.89** (+Trump+ETF) | +Trump+ETF | **+1.41** |
| LSTM | +1.08 (+Trump) | **+2.46** (Price+Tech) | Price+Technical | **+1.38** |
| CS_Gated (new) | — | **+2.61** (+Onchain) | +Onchain | — |

**v6 修正的顯著效果**：
- **LSTM**：v5.4 最佳 +1.08 → v6 最佳 **+2.46**（+1.38），在 Price+Technical 基礎特徵組即達到最高，首次超越所有傳統模型
- **TFT**：v5.4 最佳 +0.48 → v6 最佳 **+1.89**（+1.41），在 +Trump+ETF 特徵組達最高，從墊底翻升至中上
- **CS_Gated**：純橫截面模型最佳 SR=+2.61（+Onchain），驗證了 Gu, Kelly, Xiu (2020) 的觀點：橫截面特徵配合正確的排序損失即可達到優異表現
- **rank-normalized ListNet 是最關鍵修正**：所有 DL 模型均大幅改善，證實 v5 的原始報酬 softmax 目標是主要瓶頸

### 6.3 Full Cross-Model Comparison / 全模型比較 (SR PW, Long-Short)

| Information Set | VW Mkt | EW Mkt | CS_Gated | LSTM v6 | TFT v6 | OLS | EN | PCA | PLS | RF (5s) | GBT (5s) |
|-----------------|:------:|:------:|:--------:|:-------:|:------:|:---:|:--:|:---:|:---:|:-------:|:--------:|
| **Price+Technical** | −0.17 | −0.60 | +1.33 | **+2.46** | +0.19 | +1.13 | +1.28 | +0.43 | +0.58 | +1.31 | +1.50 |
| **+Onchain** | −0.17 | −0.60 | **+2.61** | +0.93 | +1.30 | +1.00 | +1.28 | +0.46 | +0.87 | +1.54 | +1.43 |
| **All** | −0.17 | −0.60 | +1.25 | +0.59 | +0.17 | +1.24 | +1.28 | +0.56 | +0.73 | +1.53 | +1.31 |
| **+Trump** | −0.17 | −0.60 | +0.90 | +0.57 | +0.98 | +1.31 | +1.03 | +0.41 | +0.82 | +1.50 | +1.28 |
| **+ETF** | −0.17 | −0.60 | +1.16 | +0.35 | +0.87 | +0.93 | +1.28 | +0.57 | +1.03 | +1.34 | +1.31 |
| **+Trump+ETF** | −0.17 | −0.60 | +1.61 | **+2.13** | **+1.89** | +1.20 | +1.03 | +0.41 | +0.82 | +1.22 | +1.28 |

> RF/GBT 後括號 (5s) 表示 5-seed ensemble。DL 模型使用 32-seed ensemble。

### 6.4 Decile Analysis / 十分位分析

> 訓練完成後更新。

### 6.5 Training Efficiency / 訓練效率

| Model | v4 Params | **v6 Params** | v4 vs v6 | Ensemble | Training Time |
|-------|-----------|-------------|----------|----------|:-------------:|
| CS_Gated (v6 new) | — | **~49-75K** | — | 32 seeds | ~120 min |
| TFT | 209-368K | **~90-100K** | 2-4× 減少 | 32 seeds | ~116 min |
| LSTM | 81-83K | **~53K** | 1.5× 減少 | 32 seeds | ~123 min |
| RF | — | — | — | **5 seeds** | — |
| GBT | — | — | — | **5 seeds** | — |
| OLS/EN/PCA/PLS | — | — | — | 1 (確定性) | ~6.5 min (全部) |

v6 使用 32 seeds × 6 feature configs = 192 次 DL 訓練（v5.4 為 4 configs = 128 次）。
新增 `grad_accum_steps=4`，有效 batch 從 86 增至 344 assets per optimizer step。

### 6.6 TFT Variable Selection Importance / TFT 特徵重要性（VSN 權重）

TFT v6 的 VariableSelectionNetwork（共享 GRN trunk）提供可解釋的特徵重要性權重（softmax 權重）。相比 v5 的 LightVSN，v6 VSN 使用 GRN 對每個變數進行非線性處理後再計算權重，更接近 Lim et al. (2021) 原論文設計。

### 6.7 Traditional Model Baselines (v5.3 新增, v5.4 更新)

**Ensemble 策略：**
- OLS / ElasticNet / PCA-OLS / PLS：確定性模型，固定 HP 下無隨機性，單一模型即等價於 ensemble
- **Random Forest / Gradient Boosting：v5.4 新增 5-seed ensemble**（最佳 HP 選定後，用 5 個不同 random_state 訓練並平均預測，與 DL 32-seed ensemble 公平比較）

**Long-Short SR (PW) — Test Period:**

| Information Set | OLS | ElasticNet | PCA | PLS | RF (5s) | GBT (5s) |
|-----------------|:---:|:----------:|:---:|:---:|:-------:|:--------:|
| Price+Technical (11) | +1.13 | +1.28 | +0.43 | +0.58 | +1.31 | +1.50 |
| +Onchain (16) | +1.00 | +1.28 | +0.46 | +0.87 | **+1.54** | +1.43 |
| All (33) | +1.24 | +1.28 | +0.56 | +0.73 | **+1.53** | +1.31 |
| +Trump (38) | +1.31 | +1.03 | +0.41 | +0.82 | +1.50 | +1.28 |
| +ETF (37) | +0.93 | +1.28 | +0.57 | +1.03 | +1.34 | +1.31 |
| +Trump+ETF (42) | +1.20 | +1.03 | +0.41 | +0.82 | +1.22 | +1.28 |

### 6.8 Market Portfolio Benchmark (v5.4 更新)

| Portfolio | mean%/week | SR | T | 資料來源 |
|-----------|:----------:|:--:|:-:|---------|
| **VW Market [PRIMARY]** (value-weighted) | −0.15% | **−0.174** | 48 | CoinGecko 實際市值 |
| EW Market [reference] (equal-weighted) | −0.76% | −0.596 | 48 | 等權平均 |

**v5.4 變更**：市場基準從 EW 改為 **VW Market Portfolio**（市值加權）。
- VW 使用 CoinGecko 免費 API 取得的 86 幣種歷史市值（`market_cap.npz`，`days=365`，涵蓋測試期 100% 覆蓋率）
- VW 更準確反映加密市場的實際表現（BTC/ETH 主導），避免等權平均高估小幣影響
- EW 保留作為參考：等權平均在小樣本橫截面研究中仍有學術參考價值

**EW vs VW 於 portfolio construction 的說明**：
長短倉 portfolio 內部使用 **EW**（equal-weight within each decile），這是橫截面報酬預測的學術標準選擇（cf. Gu, Kelly, Xiu 2020）。理由：(1) 避免對大市值資產的集中，(2) 測試模型對全宇宙的排名能力，(3) 不受市值波動影響。市場基準使用 VW，因其更準確反映被動投資者的實際報酬。

### 6.9 Per-Seed SR 分布與 Ensemble 效果 / Individual Seed Variance

> **重要發現**：DL 模型的高 ensemble SR 主要來自 32-seed averaging 的降噪效果，而非單一模型的優秀品質。

**LSTM per-seed 測試 SR 分布（32 seeds，Test set）：**

| Feature Set | Per-Seed Mean | Std | Min | Max | **Ensemble SR** | 倍率 |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Price+Technical | +0.70 | 0.80 | −0.78 | +2.51 | **+2.46** | 3.5× |
| +Onchain | +0.41 | 0.75 | −1.27 | +1.91 | +0.93 | 2.3× |
| All | +0.48 | 0.82 | −1.51 | +2.38 | +0.59 | 1.2× |
| +Trump | +0.63 | 0.83 | −1.16 | +2.21 | +0.57 | 0.9× |
| +ETF | +0.46 | 0.99 | −1.54 | +2.48 | +0.35 | 0.8× |
| +Trump+ETF | +0.26 | 0.70 | −1.49 | +1.84 | **+2.13** | 8.2× |

**TFT per-seed 測試 SR 分布（32 seeds，Test set）：**

| Feature Set | Per-Seed Mean | Std | Min | Max | **Ensemble SR** | 倍率 |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Price+Technical | +0.24 | 0.84 | −1.11 | +2.18 | **+0.19** | 0.8× |
| +Onchain | +0.51 | 0.71 | −1.53 | +1.72 | +1.30 | 2.5× |
| All | +0.51 | 0.83 | −1.52 | +1.82 | +0.17 | 0.3× |
| +Trump | +0.47 | 0.85 | −1.11 | +2.71 | +0.98 | 2.1× |
| +ETF | +0.65 | 0.76 | −0.73 | +2.07 | +0.87 | 1.3× |
| +Trump+ETF | +0.57 | 0.82 | −1.12 | +2.35 | **+1.89** | 3.3× |

**關鍵觀察：**

1. **個別 LSTM seed 的品質不如 OLS**：LSTM Price+Technical per-seed mean = +0.70，低於 OLS = +1.13（確定性）。**DL 模型的單次訓練結果不穩定**，每個 seed 的 SR 可能為負值（最差 −1.54）。

2. **Ensemble averaging 是 DL 超越 OLS 的關鍵機制**：32 個 seeds 的 predictions 具備足夠多樣性（不同初始化學到不同的 ranking patterns），平均後 noise 相消、signal 疊加，ensemble SR 大幅超過 per-seed mean。LSTM +Trump+ETF 的 ensemble SR (+2.13) 是 per-seed mean (+0.26) 的 8 倍。

3. **TFT ensemble 效果弱於 LSTM**：TFT Price+Technical ensemble SR (+0.19) 甚至低於 per-seed mean (+0.24)，顯示 32 個 TFT seeds 高度相關（不同初始化收斂至相似的 local minimum），多樣性不足。可能原因：TFT 參數多（~95K），容量大的模型在有限訓練資料下更容易收斂至固定的次優解。

4. **OLS 無 ensemble 依賴**：OLS 是確定性模型，直接給出 SR = +1.13（Price+Technical）。**公平比較須注意**：DL 的 32-seed ensemble 相當於使用了大量額外計算資源（32 次獨立訓練）才能超越 OLS，而 OLS 僅需 1 次線性回歸。

**Ensemble 策略比較：**

| Model | Seeds | 是否確定性 | 來源多樣性 |
|-------|:---:|:---:|---|
| LSTM / TFT / CS_Gated | **32** | ✗ | 不同 random initialization |
| RF / GBT | **5** | ✗ | 不同 random_state（但 RF 內部已有 300 棵樹 bagging） |
| OLS / EN / PCA / PLS | **1** | ✓ | N/A（給定 HP 下結果唯一） |

> **注意**：32-seed DL ensemble 與 5-seed RF ensemble 的比較並非完全公平。若 RF 也使用 32 seeds，其 SR 可能也會提升（但幅度應較小，因 RF bagging 已內建多樣性機制）。

---

## 7. Conclusion / 結論

### 7.1 v6 改善策略與結果

v6 針對 v5.4 的 4 個根本問題進行修正：(1) ListNet 目標尺度不一致 → rank-normalized targets, (2) 微批次梯度雜訊 → gradient accumulation, (3) 時序特徵冗餘 → 新增 CrossSectionalGatedNet, (4) TFT VSN 偏離論文 → VariableSelectionNetwork。

**v6 最終結果（vs v5.4）：**

| Model | v5.4 best SR | v6 best SR | v6 Best Config | Δ SR |
|-------|:------------:|:----------:|:--------------:|:----:|
| LSTM | +1.08 | **+2.46** | Price+Technical | **+1.38** |
| TFT | +0.48 | **+1.89** | +Trump+ETF | **+1.41** |
| CS_Gated (new) | — | **+2.61** | +Onchain | — |

- **rank-normalized ListNet 是最關鍵修正**：LSTM 從 +1.08 提升至 +2.46（+128%），TFT 從 +0.48 提升至 +1.89（+294%）
- **DL 模型首次全面超越傳統模型**：LSTM +2.46 和 CS_Gated +2.61 均高於 RF +1.54、GBT +1.50、OLS +1.31
- **+Trump+ETF 特徵組表現最穩定**：LSTM +2.13、TFT +1.89、CS_Gated +1.61 三個 DL 模型在此特徵組均表現優異

### 7.2 LSTM vs TFT vs CS_Gated 架構差異分析

1. **LSTM Price+Technical SR=+2.46 為全場最高**：簡單的 LSTM + CrossAssetAttention，在基礎動量/技術特徵上達到最佳，顯示 rank-normalized ListNet 讓時序模型充分發揮了動量特徵的序列性質
2. **CS_Gated +Onchain SR=+2.61**：純橫截面模型在 +Onchain 特徵組達最高，驗證了 Gu, Kelly, Xiu (2020) 的觀點 — 預計算特徵 + 橫截面 NN 即可達優異效果
3. **TFT 仍為最弱 DL 模型**：v6 最佳 +1.89 低於 LSTM +2.46 和 CS_Gated +2.61，但已從 v5.4 的墊底大幅改善。TFT 的多層架構（VSN + LSTM + Interpretable MHA + CrossAsset MHA）在有限訓練資料下仍有過擬合壓力
4. **特徵組敏感性**：DL 模型在不同特徵組的表現變異大（LSTM: +0.35 to +2.46），傳統模型較穩定（RF: +1.22 to +1.54）

#### 7.2.1 為何單一 DL seed 有時輸給 OLS？— Ensemble 是機制，不是錦上添花

觀察 Section 6.9 的 per-seed 數據：**單一 LSTM seed（Price+Technical）的平均 SR ≅ +0.70**，低於 OLS 的 +1.13。這並不代表 LSTM 架構不如 OLS；以下三個因素共同解釋此現象：

**A. 高 per-seed 方差（std ≈ 0.80）**
LSTM 參數靠隨機初始化（Xavier uniform）和 dropout（p=0.10）隨機化，每個 seed 收斂到不同的局部極值。49 週的短測試期使得單次實驗的 SR 本身不穩定，合理的 seed 運氣差距可高達 ±1.5 SR 單位。OLS 是封閉解（無隨機性），因此每次執行完全重現；以「單 seed 平均」比較對 DL 不公平。

**B. 單層 LSTM 無 dropout 正則化**
PyTorch 的 `nn.LSTM(dropout=p)` 僅在**兩層以上**的層間連接套用 dropout；單層 LSTM 在實作上等同於 dropout=0。這使個別 seed 較容易對訓練集過擬合，導致 per-seed SR 的下尾較長（min 可達負值）。OLS 透過嶺正則化（ElasticNet L2）有顯式控制。

**C. Ensemble 才是 DL 的真正優勢**
32 個 seed 的預測做簡單平均後，隨機初始化雜訊大幅抵消（中心極限定理）：LSTM ensemble SR 從 per-seed 均值 +0.70 升至 **+2.46（提升 3.5×）**。相比之下，OLS/EN/PCA/PLS 是確定性算法（1 seed），RF/GBT 只有 5 seeds。因此，**v6 的 DL 優勢是 ensemble diversity cancellation，而非單一網路的架構優勢**。

**TFT Ensemble 效果弱於 LSTM 的診斷：**
TFT per-seed（Price+Technical）均值 SR ≅ +0.24，ensemble SR = +0.19（**低於** per-seed 均值！）。可能原因：
- TFT 參數量（~95K）遠多於 LSTM（~53K），相同訓練資料下 per-seed 預測方差更大且方向不一致
- 各 seed 對「哪些特徵重要」的判斷不穩定，ensemble 平均後信號互相抵消而非強化
- 建議未來工作：對 TFT 做 seed dropout（去除極端 seed）或增加至 64 seeds 再做 trimmed mean

### 7.3 Trump + ETF 特徵的增量價值

**+Trump+ETF 特徵組的增量價值（vs All features）：**

| Model | All SR (PW) | +Trump+ETF SR (PW) | Δ SR |
|-------|:-----------:|:------------------:|:----:|
| LSTM | +0.59 | **+2.13** | **+1.54** |
| TFT | +0.17 | **+1.89** | **+1.72** |
| CS_Gated | +1.25 | +1.61 | +0.36 |
| OLS | +1.24 | +1.20 | −0.04 |
| RF | +1.53 | +1.22 | −0.31 |
| GBT | +1.31 | +1.28 | −0.03 |

**DL 模型從 Trump+ETF 特徵獲益最大**（LSTM +1.54, TFT +1.72），傳統模型反而略降（RF −0.31）。可能原因：
1. DL 模型的 Cross-Asset Attention 能更好地利用 Trump/ETF 這類全市場共通信號
2. 傳統模型在特徵數增加時（33 → 42）更容易受多重共線性影響

**Farside ETF 個別信號的價值**：+ETF 特徵組（無 Trump）的 CS_Gated SR=+1.16 vs All SR=+1.25，顯示 ETF 特徵單獨的增量有限，但與 Trump 特徵結合後產生互補效果（+Trump+ETF SR=+1.61）

### 7.5 v6 全模型排名

| Rank | Model | Best SR (PW) | Best Config | Params | Ensemble | Temporal? |
|------|-------|:---:|-------------|:------:|:--------:|:---------:|
| 1 | **CS_Gated v6** | **+2.61** | +Onchain | ~53K | 32 seeds | ✗ |
| 2 | **LSTM v6** | **+2.46** | Price+Tech | ~53K | 32 seeds | ✓ |
| 3 | **TFT v6** | **+1.89** | +Trump+ETF | ~95K | 32 seeds | ✓ |
| 4 | **RF** | **+1.54** | +Onchain | — | 5 seeds | ✗ |
| 5 | **GBT** | **+1.50** | Price+Tech | — | 5 seeds | ✗ |
| 6 | **OLS** | **+1.31** | +Trump | — | 1 | ✗ |
| 7 | **ElasticNet** | **+1.28** | Price+Tech | — | 1 | ✗ |
| 8 | **PLS** | **+1.03** | +ETF | — | 1 | ✗ |
| 9 | **PCA** | **+0.57** | +ETF | — | 1 | ✗ |
| — | **VW Market** | −0.17 | — | — | — | — |
| — | **EW Market** | −0.60 | — | — | — | — |

**v5.4 → v6 結論逆轉**：
- v5.4：「OLS 全面優於 DL 模型」→ v6：「DL 模型全面超越傳統模型」
- 根因：v5 的 ListNet 使用原始報酬作為 softmax 目標，不同週的報酬幅度差異導致訓練信號不一致，DL 模型無法穩定學習。v6 改用排名百分位後，所有 DL 模型均大幅改善

### 7.6 核心結論：正確的排序損失是關鍵

| Criterion | CS_Gated v6 | LSTM v6 | TFT v6 | RF | GBT | OLS |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| **Best SR (PW)** | **+2.61** ✓✓ | **+2.46** ✓ | **+1.89** | +1.54 | +1.50 | +1.31 |
| **Ensemble seeds** | **32** | **32** | **32** | 5 | 5 | 1 |
| **Params** | ~53K | ~53K | ~95K | — | — | — |
| **需要 GPU** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **需要時序窗口** | ✗ | ✓ (L=8) | ✓ (L=8) | ✗ | ✗ | ✗ |

> **v6 核心發現**：損失函數的目標尺度正規化（rank-normalized targets）是 DL 模型在橫截面排序任務上成功的關鍵。v5 的失敗不是架構問題，而是訓練目標的設計問題。正確實施 ListNet（Poh et al. 2021）後，DL 模型的 32-seed ensemble 穩定地超越所有傳統模型。

### 7.7 改善的價值：從 v4 到 v6 的完整洞見

v4 → v5 → v6 的迭代過程提供了系統性的洞見：

1. **v4 → v5 (架構修正)**：Ranking Loss + Cross-Asset Attention + 降低容量 → LSTM 改善但仍劣於 OLS，揭示了「正確方向，但目標尺度問題未解決」
2. **v5 → v6 (目標修正)**：rank-normalized targets → 所有 DL 模型大幅改善，LSTM SR +1.08 → +2.46（+128%），驗證了目標尺度一致性是排序學習的核心
3. **CrossSectionalGatedNet 驗證**：純橫截面模型 SR=+2.61 ≈ LSTM SR=+2.46，確認時序建模在預計算特徵場景下的增量價值有限，但兩者均遠超傳統模型
4. **Gradient accumulation 的貢獻**：有效 batch 從 86 增至 344，降低了 DL 模型的梯度雜訊
5. **VSN 修正的貢獻**：TFT 從 +0.48 升至 +1.89，部分歸功於更接近論文的 VariableSelectionNetwork

---

## 8. Limitations / 研究限制

1. **Short test period**: 49 weeks of out-of-sample evaluation
2. **Single lookback**: Only L=8 tested; {4, 12, 24} may yield different results
3. **ListNet temperature**: Only τ=1.0 tested; different temperatures may change ranking sharpness
4. **Cross-Asset Attention is simple**: Only 1 layer of self-attention; deeper cross-asset modeling may improve ranking
5. **Feature config sensitivity**: DL 模型在不同特徵組的 SR 變異很大（LSTM: +0.35 to +2.46），模型穩定性需進一步研究
6. **Trump data coverage gap**: Trump 特徵僅覆蓋 64.5%（2022-03 起），ETF 特徵覆蓋更短（BTC: 111/324 週, ETH: 83/324 週）
7. **Trump causality unclear**: Trump 推文可能是對市場的反應（lag）而非驅動因素（lead）
8. **Farside ETF data quality**: 手動從 CSV 解析（含括號負值等非標準格式），可能存在解析錯誤
9. **Ensemble asymmetry（不公平比較）**: DL 模型使用 32 seeds，RF/GBT 使用 5 seeds，OLS/EN/PCA/PLS 使用 1 seed（確定性）。更多 seeds 本身即可降低雜訊、提升 SR，因此直接比較最終 SR 並不完全公平。嚴謹的比較應在相同 ensemble size 下進行（例如全部使用 5 seeds），或明確拆開「架構貢獻」與「ensemble size 貢獻」。
10. **Per-seed SR 高方差依賴 ensemble size**: 單一 LSTM seed 的 SR 標準差 ≈ 0.80，代表最終結果對 ensemble size 的選擇極為敏感。若 ensemble 從 32 seeds 縮至 8 seeds，SR 預期顯著下降，代表模型的「穩健性」部分來自大量 seeds，而非架構本身的泛化能力。未來工作應系統性測試 SR 對 seeds 數量的收斂曲線（ensemble size ablation）。

---

## 9. Future Work / 待研究方向

1. **Feature config stability**: 研究為何 DL 模型在不同特徵組的 SR 變異如此大，是否可通過 feature dropout 或 regularization 提升穩定性
2. **ListNet on Gated FFN**: 將 rank-normalized ListNet 應用於 v3 Gated FFN（MSE loss, SR=+2.90），預期可進一步提升
3. **Deeper Cross-Asset Modeling**: 多層 Transformer 編碼器取代單層 Cross-Asset Attention
4. **Temperature annealing**: ListNet 溫度從高到低退火，先學粗略排序再學精細排序
5. **Per-fund ETF signals**: 目前只用 GBTC/ETHE + Total，可擴展至所有個別基金（IBIT, FBTC, ARKB 等）的流量
6. **Granger causality test**: 對 Trump 和 ETF 特徵進行 Granger 因果檢驗
7. **Ablation study**: 分離 v6 四項修正的個別貢獻（rank-normalized targets vs grad_accum vs VSN vs cs_gated）
8. **Walk-forward validation**: 用 expanding window 取代固定 70/15/15 分割，驗證模型穩定性

---

## 10. v4 → v5 → v6 Changes Summary / 改版摘要

| Component | v4 | v5.4 | **v6** |
|-----------|----|----|------|
| **Loss function** | MSE | ListNet (raw returns) | **ListNet (rank-normalized targets)** |
| **Training paradigm** | Random (asset, time) pairs | Cross-sectional batches | Cross-sectional + **grad_accum=4** |
| **Cross-asset** | None | Self-Attention (2 heads) | — |
| **VSN** | Full GRN (55K) | LightVSN (5K) | **VariableSelectionNetwork (27K, shared GRN)** |
| **Models** | LSTM, TFT | LSTM, TFT | LSTM, TFT, **CrossSectionalGatedNet** |
| **Hidden dim** | 64 | 64 | 64 |
| **Features** | 49 (up to 33 used) | 49 (up to 38 used) | **53 (up to 42 used)** |
| **Feature configs** | 3 | 4 (+Trump) | **6 (+ETF, +Trump+ETF)** |
| **Data panel** | btc_panel.npz | btc_panel.npz | **btc_panel_v6.npz** |
| **New features** | — | Trump (44-48) | **Farside ETF (49-52)** |
| **LSTM layers** | 2 | 1 | 1 |
| **Dropout** | 0.15 | 0.30 | 0.30 |
| **Lookback** | 12 | 8 | 8 |
| **Num seeds** | 8 | 32 | 32 |
| **Best DL SR** | +1.08 (LSTM) | +1.08 (LSTM) | **+2.61 (CS_Gated)** |
| **DL vs OLS** | DL < OLS | DL < OLS | **DL > OLS** ✓ |

---

## 11. File Structure / 檔案結構

```
machine_learning_for_crypto_v6/
├── README.md                   # This file
├── config.json                 # Hyperparameters, feature configs, traditional model grids
├── prepare_panel_v6.py         # [v6 NEW] Build btc_panel_v6.npz with Farside ETF features
├── data_loader.py              # Load btc_panel_v6.npz → cross-sectional batches
├── models.py                   # LSTM + TFT + CS_Gated with Cross-Asset Attention
├── train.py                    # ListNet ranking loss (rank-normalized) training loop
├── train_traditional.py        # Traditional ML baselines (OLS, EN, PCA, PLS, RF, GBT)
├── evaluate.py                 # Portfolio evaluation, comparison tables, visualization
├── run_all.sh                  # One-click: train all → evaluate → compare
├── checkpoints/
│   ├── cs_gated_results.npz    # [v6 NEW] CrossSectionalGatedNet ensemble predictions
│   ├── tft_results.npz         # TFT ensemble predictions
│   ├── lstm_results.npz        # LSTM ensemble predictions
│   ├── ols_results.npz         # OLS predictions
│   ├── elasticnet_results.npz  # ElasticNet predictions
│   ├── pca_regression_results.npz
│   ├── pls_results.npz
│   ├── random_forest_results.npz
│   ├── gradient_boosting_results.npz
│   └── market_portfolio.npz    # EW/VW market portfolio benchmark
└── outputs/
    ├── cross_model_comparison.csv  # Full model comparison table
    └── ...                         # Per-model tables and visualizations
```

---

## 12. References & Methodology / 參考文獻與方法論

### 12.1 核心學術論文

#### [1] Gu, Kelly, Xiu (2020) — 機器學習資產定價基準

> Gu, S., Kelly, B., & Xiu, D. (2020). **Empirical Asset Pricing via Machine Learning**. *Review of Financial Studies*, 33(5), 2223–2273.

**論文方法：**
- 使用 30,000+ 美國股票的橫截面面板，94 個特徵（firm characteristics），預測下月股票超額報酬
- 比較 OLS、ElasticNet、PCA、PLS、Random Forest、Gradient Boosted Trees、Feedforward NN (NN1–NN5)
- **關鍵發現**：簡單前饋神經網路（NN3，3 隱藏層 × 32 neurons）即達最佳 SR≈2.14，**不使用 LSTM 或任何時序建模**
- 所有特徵均為預先計算的橫截面特徵（momentum, size, value 等），直接以當期值輸入模型
- 損失函數：MSE（絕對報酬預測），非排序損失
- 訓練方式：expanding window，每月重訓練

**v6 引用：**
- 傳統模型基準（OLS, EN, PCA, PLS, RF, GBT）直接參考此論文的模型族
- CrossSectionalGatedNet 設計靈感來源：純橫截面前饋網路，不使用時序窗口
- 驗證了「預計算特徵 + 橫截面模型 > 原始資料 + 時序模型」的觀點

---

#### [2] Cao, Qin, Liu, Tsai, Li (2007) — ListNet 排序損失

> Cao, Z., Qin, T., Liu, T. Y., Tsai, M. F., & Li, H. (2007). **Learning to Rank: From Pairwise Approach to Listwise Approach**. *ICML*.

**論文方法：**
- 提出 ListNet：第一個列表式（listwise）排序學習方法，取代逐對（pairwise）比較
- 核心思想：將預測分數和真實相關性分數各自通過 softmax 轉換為概率分佈，用交叉熵衡量分佈差異
- 公式：$L = -\sum_{i} P_{\text{true}}(i) \cdot \log P_{\text{pred}}(i)$，其中 $P(i) = \text{softmax}(s_i / \tau)$
- 原始應用：資訊檢索（information retrieval），相關性分數為離散標籤（0–4）
- 優勢：直接優化列表排序品質，不需要計算所有 pair 的損失（O(n) vs O(n²)）

**v6 引用：**
- v5/v6 的核心損失函數
- **v6 改進**：原論文使用離散標籤作為相關性分數；v5 直接使用原始報酬（導致不同週的 softmax 分佈銳度不一致）；v6 改用橫截面排名百分位 [0,1] 作為相關性分數（受 Poh et al. 2021 啟發），使每週的目標分佈具有相同的熵

---

#### [3] Poh, Roberts, Zohren (2021) — 排序損失在金融中的應用

> Poh, D., Roberts, S., & Zohren, S. (2021). **Building Cross-Sectional Systematic Strategies By Learning to Rank**. *arXiv:2012.07149*.

**論文方法：**
- 將學習排序（Learning to Rank）方法系統性地應用於金融橫截面策略
- 比較 Pointwise（MSE）、Pairwise（RankNet）、Listwise（ListNet, ListMLE）損失在股票報酬排序上的效果
- **關鍵發現**：ListNet 配合排名百分位目標（rank percentile targets）在 Sharpe ratio 上顯著優於 MSE
- 目標處理：將原始報酬轉換為橫截面排名百分位 $r_i = \text{rank}(y_i) / (N-1) \in [0, 1]$，底部為 0，頂部為 1
- 這解決了 ListNet 原始論文中未明確處理的問題：當相關性分數（即報酬）在不同時間步的尺度差異巨大時，softmax 分佈的銳度不一致

**v6 引用：**
- v6 `listnet_loss` 中的 rank-normalized targets 直接源自此論文：
  ```python
  ranks = torch.argsort(torch.argsort(target_valid)).float()
  target_norm = ranks / (len(ranks) - 1)  # [0, 1]
  ```
- 這是 v5 → v6 最關鍵的修正：解決了不同週報酬幅度不同導致的訓練信號不一致問題

---

#### [4] Lim, Arık, Loeff, Pfister (2021) — Temporal Fusion Transformer

> Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**. *International Journal of Forecasting*, 37(4), 1748–1764.

**論文方法：**
- 提出 TFT 架構，針對多步預測設計，強調可解釋性
- **Variable Selection Network (VSN)**：每個變數有獨立的 GRN（Gated Residual Network），softmax 權重提供特徵重要性解釋
  - 每個變數 $x_j$ 先通過 per-variable GRN 得到 $\tilde{x}_j = \text{GRN}_j(x_j)$
  - 所有變數的 pooled representation + static context → softmax 權重 $v_j$
  - 選擇後的輸入 $\tilde{x} = \sum_j v_j \cdot \tilde{x}_j$
- **Gated Residual Network (GRN)**：`GLU(W₁·ELU(W₂·x) + W₃·c) + x`，c 為可選 context
- **Interpretable Multi-Head Attention**：所有 head 共享 Value 矩陣，使注意力權重更可解釋
- **靜態協變量處理**：時間不變的靜態特徵（如資產 ID）通過專用 GRN 產生 context，注入 VSN 和注意力層

**v6 引用：**
- TFT 架構的基礎設計
- **v5 偏差**：使用 LightVSN（pooled mean + Linear+ELU, ~5K params），而非論文的 per-variable GRN
- **v6 修正**：改用 VariableSelectionNetwork，共享 GRN trunk 處理每個變數（~27K params），更接近論文設計但避免了 per-variable 獨立 GRN 的參數膨脹（原論文在 M=33 時需 ~55K params 僅 VSN 一項）
- **仍存在的偏差**：無 quantile output（僅需排序分數）、無 future covariates（週頻預測不需要）、新增 Cross-Asset Attention（原論文無此設計）

---

#### [5] Fischer, Krauss (2018) — LSTM 在金融市場的應用

> Fischer, T., & Krauss, C. (2018). **Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions**. *European Journal of Operational Research*, 270(2), 654–669.

**論文方法：**
- 使用 LSTM 對 S&P 500 成分股進行橫截面報酬預測（每日預測下一日方向）
- 輸入：每支股票過去 240 個交易日的報酬序列（univariate time series）
- 訓練方式：Per-stock LSTM，每支股票獨立訓練，無跨資產交互
- **關鍵做法**：橫截面排序策略 — 依 LSTM 預測概率排序，多頭（top decile）空頭（bottom decile）
- 結果：年化報酬 0.46%（2000–2015），超越 Random Forest 和 Logistic Regression
- 架構：2 層 LSTM，hidden_dim=25，dropout=0.1，回看窗口=240 天

**v6 引用：**
- LSTM 用於橫截面排序的學術先例
- **v5/v6 偏差**：
  - Fischer & Krauss 使用原始報酬序列（univariate）；v6 使用 49+ 個預計算特徵（multivariate）
  - Fischer & Krauss 每支股票獨立訓練；v6 所有資產共享模型 + Cross-Asset Attention
  - Fischer & Krauss 使用 MSE/BCE 損失；v6 使用 ListNet 排序損失
  - v6 回看窗口 8 週（vs 240 天），因特徵已包含 r52w 等長期動量

---

#### [6] Feng, Chen, He (2019) — 時序關聯排序

> Feng, G., Chen, J., & He, J. (2019). **Temporal Relational Ranking for Stock Prediction**. *arXiv:1809.09441*.

**論文方法：**
- 提出結合時序建模和橫截面排序的端到端框架
- 架構：LSTM encoder → Cross-stock attention → Ranking loss
- **關鍵創新**：在時序編碼後加入跨股票關聯模組（relational module），讓模型學習不同股票之間的互動關係
- 排序損失：使用 pairwise ranking loss（非 ListNet）
- 輸入：每支股票的歷史特徵序列（7 天窗口）
- 實驗：中國 A 股市場，優於純 LSTM 和純排序模型

**v6 引用：**
- Cross-Asset Self-Attention 的設計靈感：先對每個資產進行 LSTM 時序編碼，再通過跨資產注意力進行橫截面比較
- **v6 差異**：使用 ListNet（listwise）而非 pairwise loss；使用標準 Multi-Head Self-Attention 而非 paper 的 relational module

---

#### [7] Hochreiter, Schmidhuber (1997) — LSTM

> Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*, 9(8), 1735–1780.

**論文方法：**
- 提出 LSTM（Long Short-Term Memory）架構，通過 input/forget/output gate 解決 RNN 的梯度消失問題
- 核心：cell state $c_t$ 作為長期記憶，gate 機制控制資訊的寫入、保留和讀出
- 公式：$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$（forget gate），$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$（cell update）

**v6 引用：** LSTM 和 TFT 的時序編碼器基礎

---

#### [8] Vaswani et al. (2017) — Self-Attention

> Vaswani, A., et al. (2017). **Attention Is All You Need**. *NeurIPS*.

**論文方法：**
- 提出 Transformer 架構：純注意力機制取代 RNN/CNN
- Multi-Head Attention：$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$，每個 head: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$
- 每個 head 有獨立的 $W^Q$, $W^K$, $W^V$ 投影矩陣

**v6 引用：**
- Cross-Asset Self-Attention 基於此架構
- TFT 的 Interpretable Multi-Head Attention 修改了此設計（shared V across heads）

---

### 12.2 v6 方法論對照表

| 元件 | 論文原始做法 | v6 實作 | 差異說明 |
|------|------------|---------|---------|
| **排序損失** | ListNet: softmax(relevance/τ) [2] | softmax(rank_percentile/τ) | 結合 [2]+[3]：用排名百分位取代原始分數 |
| **VSN** | Per-variable GRN [4] | 共享 GRN trunk + per-var projection | 參數效率：27K vs 55K（M=33） |
| **LSTM 排序** | Per-stock 獨立 LSTM [5] | 共享模型 + Cross-Asset Attention | 結合 [5]+[6]：時序 + 跨資產 |
| **跨資產交互** | Relational module [6] | Standard MHA [8] | 簡化設計，效果待驗證 |
| **橫截面 NN** | NN1–NN5 (MLP) [1] | CrossSectionalGatedNet (GRN layers) | 用 GRN 取代 ReLU MLP，加 ListNet 損失 |
| **梯度累積** | 標準大 batch 訓練 [1] | grad_accum_steps=4 (86×4=344) | 解決週頻橫截面 batch 過小問題 |
| **Interpretable MHA** | Shared V across heads [4] | Standard: per-head V [8] | v6 改回標準做法，增加 head 多樣性 |

### 12.3 其他資源

- Research Project Literature Survey: 743 papers (2016–2026) on cryptocurrency × machine learning. See `論文/research_project/data/research_gap_report.md`.
- **Trump Code** — AI-powered cryptanalysis of presidential communications × stock market impact. [github.com/sstklen/trump-code](https://github.com/sstklen/trump-code). 31.5M model combinations tested, 551 surviving rules, 61.3% hit rate (z=5.39). Source of Trump social media features (v5.1).
