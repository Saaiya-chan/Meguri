# Phase 2 シミュレーション 進捗状況レポート

**更新日：** 2026-02-15（Stage 3 完了）
**詳細レポート：** [`simulation/results/final_report.md`](results/final_report.md)

---

## 1. 実装状況

### ✅ 実装済み（Stage 1 〜 3 全完了）

| モジュール | ファイル | 内容 | Stage |
|---|---|---|---|
| ネットワーク生成 | `simulation/src/network_generator.py` | Watts-Strogatz小世界ネットワーク | 1 |
| 取引シミュレーション | `simulation/src/transaction_engine.py` | Poisson(λ=2) 日次取引、攻撃者戦略3種 | 1 |
| スコアリング | `simulation/src/scoring.py` | Bell-Curve T(v)、Partner-Diversity E(v)、D(v)、softmin | 1+**3改** |
| Mana配分 | `simulation/src/mana_distributor.py` | 75%均等 + 25% VRF Bloom、DecayEngine | 2 |
| コミュニティ検出 | `simulation/src/community_detector.py` | Louvain月次、α(c)正規化 | 2 |
| 評価指標 | `simulation/src/metrics.py` | Sybil ROI, TPR/FPR, Gini, VRF公平性 | 2 |
| 実行基盤 | `simulation/src/runner.py` | 180日ループ、全シナリオ、距離軸オプション | 2+**3改** |
| DoEオプティマイザ | `simulation/src/doe_optimizer.py` | L16直交表、16実験自動実行 | **3新** |
| T010スクリプト | `simulation/scenario_t010.py` | 3層 vs 4層 距離軸比較 | **3新** |
| Notebook（Stage 1） | `simulation/notebooks/01_basic_simulation.ipynb` | MVP検証、スコア分布 | 1 |
| Notebook（Stage 2） | `simulation/notebooks/02_scenario_testing.ipynb` | T001〜T010実行、全指標可視化 | 2 |
| Notebook（Stage 3） | `simulation/notebooks/03_parameter_optimization.ipynb` | DoE分析、感度分析、T010比較 | **3新** |

### ❌ 未実施（Phase 3 以降）

| 項目 | 優先度 | 理由 |
|---|---|---|
| T004実行（N=100K） | 中 | 推定4〜6時間。大規模マシン/クラウドで実行を推奨 |
| D軸（デバイス紐づけ）本実装 | **高** | ZK-SNARKまたは類似証明でSybil ROI < 0.1 達成が前提 |
| ループ攻撃対策追加 | 高 | 日次取引上限（cap）の導入でROI(C5)を劇的改善できる見込み |
| Gini 改善 | 中 | Bloom 受賞者を複数分散（1巣 → 3〜5巣）で改善期待 |
| ZK-SNARK 証明コスト確認 | 中 | シミュレーション外の実装・計測が必要 |

---

## 2. テストケース実行状況

### Stage 2（デフォルトパラメータ）

**実行条件：** κ=1.0, θ=0.3, β=1.0, Bloom=7日, Grace=30日

| ID | シナリオ | N | k | Sybil ROI | 目標 | TPR | 目標 | FPR | 目標 | Gini |
|---|---|---|---|---|---|---|---|---|---|---|
| T001 | A: 単純Sybil | 1K | 1 | 0.250 | <1.000 ✅ | 0% | >95% ❌ | 12.01% | <1% ❌ | 0.685 ❌ |
| T002 | A: 単純Sybil | 1K | 10 | 0.645 | <0.100 ❌ | 10% | >95% ❌ | 7.78% | <1% ❌ | 0.702 ❌ |
| T003 | A: 単純Sybil | 10K | 50 | 0.556 | <0.020 ❌ | 14% | >95% ❌ | 9.28% | <1% ❌ | 0.704 ❌ |
| T004 | A: 単純Sybil | 100K | 100 | — | — ⏭️ | — | — ⏭️ | — | — ⏭️ | — |
| T005 | B: デバイス制約 | 10K | 10 | 2.052 | <0.100 ❌ | 10% | >95% ❌ | 9.34% | <1% ❌ | 0.707 ❌ |
| T006 | C: 高頻度ループ | 10K | 5 | 69.745 | <0.200 ❌ | 0% | >95% ❌ | 0.01% | <1% ✅ | 0.520 ❌ |
| T007 | D: 農村FP | 10K | 0 | — | ✅ | — | ✅ | 0.17% | <1% ✅ | 0.601 ❌ |
| T008 | E: 新規参加 | 10K | 0 | — | ✅ | — | ✅ | 0.01% | <1% ✅ | 0.507 ❌ |
| T009 | F: Grace Period | 10K | 0 | — | ✅ | — | ✅ | 0.00% | <1% ✅ | 0.509 ❌ |

### Stage 3（DoE L16 実験）

**実験条件：** N=1000, k=10 (Scenario A), k=5 (Scenario C), 180日

| 順位 | κ | θ | β | Bloom | スコア | ROI(A) | FPR(A) | TPR(A) | ROI(C) |
|---|---|---|---|---|---|---|---|---|---|
| **1位（推奨）** | **0.5** | **0.35** | **0.5** | **7d** | **22.28** | 0.679 | 28.1% | 30.0% | 70.9 |
| 2位 | 2.5 | 0.60 | 2.0 | 7d | 26.58 | 0.608 | 100% | 100% | 68.2 |
| 3位 | 0.5 | 0.60 | 2.0 | 7d | 26.83 | 0.666 | 99.5% | 100% | 71.2 |

### Grace Period スキャン（推奨パラメータで）

| Grace Period | ROI | FPR | コミュニティ安定性 |
|---|---|---|---|
| 14日 ← **推奨** | 0.0 | 0.0% | 100% |
| 30日 | 0.0 | 0.0% | 90.3% |
| 60日 | 0.0 | 0.0% | 96.6% |

### T010 距離軸比較（Scenario G）

| k | 3層 ROI | 4層 ROI | 判定 |
|---|---|---|---|
| 1 | 0.0045 | 0.3510 | 3層 ✅ |
| 5 | 0.2632 | 0.2677 | 3層 ✅ |
| 10 | 0.5704 | 0.6647 | 3層 ✅ |
| 20 | 1.3758 | 0.9597 | 4層 ✅ |
| 50 | 0.6810 | 0.8441 | 3層 ✅ |

**→ Phase 2 は 3層（D軸なし）で継続**

---

## 3. 推奨パラメータ（Stage 3 確定値）

| パラメータ | 推奨値 | 選定根拠 |
|---|---|---|
| κ（減価加速係数） | **0.5** | DoE L16 最適解 |
| θ（異常閾値） | **0.35** | DoE L16 最適解（FPR最小化） |
| β（softmin スケール） | **0.5** | DoE L16 最適解 |
| Bloom 間隔 | **7日** | DoE L16 最適解 |
| Grace Period | **14日** | スキャン結果（最短を推奨） |
| D軸 | **不使用** | T010 で有効性不一致 |

---

## 4. Phase 2 チェックリスト 最終状況

| 項目 | 状態 |
|---|---|
| Sybil ROI(100) < 1/150 | ❌ D軸なし・T004未実行のため未達 |
| False Positive 率 < 1% | △ Scenario D/E/F は達成。Scenario A は FPR=28% |
| κ, θ の最適値を決定 | ✅ **κ=0.5, θ=0.35** |
| Bloom 間隔・配分量を決定 | ✅ **7日間隔, 75/25 比率** |
| Grace Period 期間を決定 | ✅ **14日** |
| 距離軸（D軸）の要否を結論 | ✅ **Phase 2 では不使用（3層）** |
| コミュニティ検出アルゴリズムを選定 | ✅ Louvain（安定性 90〜100%） |
| ZK-SNARK 証明コスト確認 | ❌ 未実施 |
| 全テストケースで合格 | ❌ Sybil ROI / TPR / Gini が未達 |

---

## 5. ファイル構成（Stage 3 完了後）

```
Meguri_pre3/
├── simulation/
│   ├── config.py
│   ├── network_generator.py
│   ├── transaction_engine.py
│   ├── scoring.py              ← Stage 3 強化（Bell-Curve T, Partner E, D軸）
│   ├── mana_distributor.py
│   ├── community_detector.py
│   ├── metrics.py
│   ├── runner.py               ← Stage 3 強化（use_network_distance 対応）
│   ├── doe_optimizer.py        ← Stage 3 新規
│   └── scenario_t010.py        ← Stage 3 新規
├── notebooks/
│   ├── 01_basic_simulation.ipynb
│   ├── 02_scenario_testing.ipynb
│   └── 03_parameter_optimization.ipynb ← Stage 3 新規
├── results/
│   ├── summary/                ← T001〜T009 個別 JSON
│   ├── doe/
│   │   ├── doe_l16_results.json    ← DoE 16実験結果
│   │   ├── grace_period_scan.json  ← Grace Period スキャン
│   │   └── t010_distance_axis.json ← T010 比較結果
│   └── final_report.md         ← 最終推奨パラメータレポート
├── SIMULATION_STATUS.md        ← このファイル
├── requirements.txt
└── phase2_specification.html
```
