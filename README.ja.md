# PyQCA

**PyQCA** は、Python ネイティブな **Qualitative Comparative Analysis (QCA; 質的比較分析)** のための実験的ライブラリです。

[![CI](https://github.com/t-yamsaki/PyQCA/actions/workflows/ci.yml/badge.svg)](https://github.com/t-yamsaki/PyQCA/actions/workflows/ci.yml)
[![Docs build](https://github.com/t-yamsaki/PyQCA/actions/workflows/docs.yml/badge.svg)](https://github.com/t-yamsaki/PyQCA/actions/workflows/docs.yml)
[![Documentation](https://readthedocs.org/projects/pyqca/badge/?version=latest)](https://pyqca.readthedocs.io/en/latest/)

csQCA、mvQCA、fsQCA、混合型QCAに加え、XGBoostによる条件選択と
閾値提案をPyQCAのQCA engineへ接続するmachine-learning-enhanced QCA
（mlQCA）をPythonのデータ分析ワークフロー内で扱うことを目的としています。
閾値・キャリブレーション感度分析、レポート、再現性情報にも対応します。

> ステータス: experimental / active development

---

## ドキュメント

Sphinxドキュメントには、インストール、Core/gsQCA、calibration、
minimization、感度分析、レポート、mlQCA、API referenceを収録しています。
Read the Docsで公開できる構成です。

ローカルでビルドする場合:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Read the Docs projectを接続するまでは、ビルド後に
`docs/_build/html/index.html` を開いて確認できます。

---

## 特徴

* **Core QCA**: `CSQCA`, `MVQCA`, `FSQCA`
* **GSQCA**: crisp / multi-value / fuzzy 条件を統一スキーマで扱う高レベルインターフェース
* **Minimization backends**: `standard`, `qmc`, `set_cover`, `greedy_set_cover`, `exact_set_cover`
* **Calibration**: crisp calibration, piecewise fuzzy calibration, logistic fuzzy calibration
* **ThresholdSweep**: ThSQCA に準拠した outcome / condition / grid / dual threshold sweep
* **fsQCA Anchor Sensitivity**: full membership, crossover, full non-membership anchor sweep
* **Reporting**: Markdown, LaTeX, reproducibility metadata, experiment logging, Jupyter-friendly summary
* **Visualization**: truth table, XY plot, coverage, configuration chart などの optional Matplotlib 可視化
* **mlQCA**: XGBoost による条件選択、閾値抽出、QCA探索、Pareto評価、CV/bootstrap安定性検証

---

## インストール

このブランチは **0.3.0** の公開準備版です。[変更履歴](https://pyqca.readthedocs.io/en/latest/release_notes.html)に解析結果への影響と移行時の注意点をまとめています。

公開前、または開発版をローカルで利用する場合:

```bash
git clone https://github.com/t-yamsaki/PyQCA.git
cd PyQCA
pip install -e .
```

可視化機能も使う場合:

```bash
pip install -e ".[viz]"
```

mlQCA 機能を使う場合:

```bash
pip install -e ".[mlqca]"
```

外部SHAPパッケージも含める場合:

```bash
pip install -e ".[mlqca-explain]"
```

PyPIに0.3.0が公開された後:

```bash
python -m pip install "pyqca==0.3.0"
```

---

## クイックスタート

### csQCA

```python
import pandas as pd
from qca import CSQCA

df = pd.DataFrame({
    "case": ["c1", "c2", "c3", "c4"],
    "A": [1, 1, 0, 0],
    "B": [1, 0, 1, 0],
    "Y": [1, 1, 0, 0],
})

model = CSQCA(
    data=df,
    case_id="case",
    outcome="Y",
    conditions=["A", "B"],
)

result = model.fit(consistency_cutoff=0.8)

print(result.truth_table)
print(result.solutions)
print(result.formula)
```

### fsQCA

区分線形校正の帰属度上下限は有限値で、`0 <= lower < 0.5 < upper <= 1` を満たす必要があります。クロスオーバーで片側の区間が一定値になるのを避けるため、上下限がちょうど0.5の指定は拒否します。端点0と1は使用できます。スカラー・Series・分位点に基づく区分線形校正で同じ検証を行い、不正な上下限には `ValueError` を返します。ロジスティック校正の、より厳密な端点制約は維持します。

区分線形校正は、アンカー間隔に固定の下限を設けず、実際の正の間隔で補間します。スカラー／Seriesの直接呼び出しには有限かつ厳密に昇順のアンカーが必要で、退化した指定は `ValueError` になります。分位点校正では、推定アンカーが無効なグループを警告してスキップする既存の方針を維持します。単位変換後もアンカーが有限かつ区別可能なら、浮動小数点精度の範囲で帰属度を維持します。丸めによって同一値になったアンカーは区別できず、人工的な間隔への置換は行いません。

グループ別の `calibrate_piecewise` 校正は、グループ内・グループ間でインデックスラベルが重複していても、各グループの実際の行だけを使用します。返却結果は入力の行順とインデックスを維持し、アンカーの件数・`min_nonnull`・校正済み／スキップ件数はラベルの種類数ではなく実際の行数に基づきます。

```python
import pandas as pd

from qca import FSQCA, calibrate_piecewise

fs_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 9)],
        "raw_score": [1, 2, 3, 4, 6, 7, 8, 9],
        "B": [0, 0, 0, 1, 0, 1, 1, 1],
        "Y": [0.05, 0.10, 0.20, 0.45, 0.40, 0.75, 0.85, 0.95],
    }
)

calibrated = calibrate_piecewise(
    fs_df,
    value_col="raw_score",
    out_col="A",
    min_nonnull=5,
).df

model = FSQCA(
    data=calibrated,
    case_id="case",
    outcome="Y",
    conditions=["A", "B"],
    condition_types={"A": "fuzzy", "B": "crisp"},
)

result = model.fit(consistency_cutoff=0.8, coverage_cutoff=0.1)
```

### GSQCA

真理表のグループ化には、入力列と衝突しない一時列名を使用します。`__crisp_` や `__gsqca_corner_` で始まる条件名も、事例の割当てや最小化結果を変えることなく使用できます。

`value_columns` の明示的な別名指定は、変換先へ書き込む前に元の入力から一括で解決します。列の入れ替え、参照元の共有、別条件の明示的マッピングが参照する列の利用を許容し、辞書や条件の順序に依存しません。変換先が結果列・ケースID・条件名・他の値別列（名前から暗黙的に解決する値を含む）と衝突する場合は `ValueError` になります。同じ値に対する通常の命名規則の列より明示的指定を優先します。参照元はすべて元の入力に存在する必要があり、呼び出し元のDataFrameは変更しません。

多値条件のカテゴリ値には、内部ワイルドカード用の予約値である文字列 `*` は使用できません。観測データ、宣言したドメイン、`value_columns` のキー、カテゴリ和集合にこの値が含まれる場合は `ValueError` になります。解析前に別のカテゴリ値へ再符号化してください。`A*` のように文字列の一部に含まれる場合は使用できます。

```python
import pandas as pd

from qca import GSQCA

gsqca_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 9)],
        "policy": [1, 1, 1, 0, 1, 0, 0, 0],
        "region": ["north", "south", "north", "south"] * 2,
        "capacity": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        "Y": [0.95, 0.85, 0.80, 0.65, 0.45, 0.30, 0.20, 0.10],
    }
)

model = GSQCA(
    data=gsqca_df,
    case_id="case",
    outcome="Y",
    conditions=["policy", "region", "capacity"],
    condition_types={
        "policy": "crisp",
        "region": "multi",
        "capacity": "fuzzy",
    },
)

result = model.fit(minimizer="set_cover")
```

`GSQCA` は PyQCA が文書化する操作的定義に基づく generalized-set
workflow です。特定の外部 gsQCA パッケージの完全互換クローンとしては
位置づけていません。

統一スキーマからも作成できます:

```python
schema = [
    {"name": "policy", "type": "crisp", "domain": [0, 1], "calibrated": True},
    {"name": "region", "type": "multi-value", "domain": ["north", "south"]},
    {"name": "capacity", "type": "fuzzy", "domain": [0, 1], "calibrated": True},
]

model = GSQCA.from_schema(
    data=gsqca_df,
    outcome="Y",
    schema=schema,
    case_id="case",
)
```

---

## Minimization Backends

方向指定 `directional_expectations` のキーには既存の条件名、値にはその条件のドメイン内の値（クリスプ・ファジー条件は0/1、多値条件は宣言済みまたは観測済みのカテゴリ）を指定します。未知の条件名・ドメイン外の値は、正の行がない場合も最小化前に `ValueError` になります。部分指定では省略した条件を制約しません。`None` は従来どおり警告を出して中間解を複雑解と同じにし、空辞書 `{}` は安全な論理残余に方向制約を適用しません。条件を制約しない場合は値に `None` を入れず、キーを省略してください。

`fit(minimizer=...)` で最小化バックエンドを切り替えられます。

```python
result = model.fit(minimizer="standard")
result = model.fit(minimizer="qmc")
result = model.fit(minimizer="set_cover")
result = model.fit(minimizer="greedy_set_cover")
result = model.fit(minimizer="exact_set_cover")
```

同じモデルでバックエンドを比較する場合:

```python
from qca import benchmark_minimizers

bench = benchmark_minimizers(
    model,
    minimizers=["standard", "qmc", "greedy_set_cover", "exact_set_cover"],
    outcome_threshold=0.75,
    consistency_cutoff=0.75,
)

print(bench.summary())
```

---

## ThresholdSweep

閾値二値化（`qca_bin`）は浮動小数点の帰属度を返し、有限値は閾値以上なら1、未満なら0にします。NaNと正負の無限大は欠損（NaN）として保持し、結果の否定や `pre_calibrated` 条件でも0/1へ補完しません。閾値には有限値が必要です。準備処理は欠損を保持し、各閾値分析は欠損を含む実行を無効にして、対象列と行数をサマリーの `error` に記録します。解析前に欠損を処理するか、明示的に対象行を除外してください。分析側では自動補完・行除外を行いません。

`ThresholdSweep` は ThSQCA のアルゴリズムに準拠し、raw variable を `x >= threshold` で二値化しながら QCA を反復実行します。

```python
import pandas as pd

from qca import ThresholdSweep

sweep_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 13)],
        "A": [9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 3, 2],
        "B": [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 2],
        "C": [9, 7, 8, 6, 7, 5, 6, 4, 5, 3, 4, 2],
        "Y": [10, 9, 9, 8, 8, 7, 7, 6, 5, 4, 3, 2],
    }
)

sweep = ThresholdSweep(
    sweep_df,
    outcome="Y",
    conditions=["A", "B", "C"],
    case_id="case",
)

res = sweep.outcome(
    thresholds=[5, 6, 7, 8],
    condition_thresholds={"A": 7, "B": 7, "C": 7},
    incl_cut=0.8,
    n_cut=1,
)

print(res.summary_df)
print(res.stability)
```

利用可能な sweep:

* `sweep.outcome()`
* `sweep.condition()`
* `sweep.multi_condition()`
* `sweep.dual()`
* `sweep.fuzzy_anchors()`

---

## fsQCA Anchor Sensitivity

`AnchorSensitivity` の辞書形式 `dir_exp` は、元の条件名と校正後名（既定は `<条件名>_fuzzy`、または指定した `out_col`）のどちらでも指定できます。元名を最小化前に校正後名へ対応付け、非対象条件の名前は維持します。同じ条件への重複指定は値が一致すれば許可し、矛盾する場合はその実行にエラーを記録します。未知名・不正な値もエラーです。スカラー・配列形式の意味と条件順は維持します。mlQCAアダプターでも、低方向の校正を含め、校正後の条件名による指定を引き続き使用できます。

```python
import pandas as pd

from qca import AnchorSensitivity

anchor_df = pd.DataFrame(
    {
        "case": [f"c{i}" for i in range(1, 11)],
        "Budget": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Capacity": [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        "Y": [0.05, 0.10, 0.20, 0.40, 0.55, 0.50, 0.70, 0.80, 0.90, 0.95],
    }
)

anchors = AnchorSensitivity(
    anchor_df,
    outcome="Y",
    conditions=["Budget", "Capacity"],
    case_id="case",
)

res = anchors.grid(
    "Budget",
    {
        "full_out": [1, 2],
        "crossover": [4, 5],
        "full_in": [8, 9],
    },
    incl_cut=0.8,
)

print(res.summary_df)
print(res.stability)
```

単一 anchor 系だけを sweep する場合:

```python
res = anchors.crossover(
    "Budget",
    values=[4, 5, 6],
    full_out=1,
    full_in=9,
)
```

---

## レポートと再現性

```python
from pathlib import Path

from qca import (
    ExperimentLogger,
    collect_reproducibility_metadata,
    generate_markdown_report,
    jupyter_summary,
    to_latex_table,
)

result = model.fit(consistency_cutoff=0.8)

Path("reports").mkdir(parents=True, exist_ok=True)

generate_markdown_report(result, path="reports/qca_result.md")
to_latex_table(result, table="solutions", path="reports/solutions.tex")

metadata = collect_reproducibility_metadata(result)
ExperimentLogger("reports/experiments.jsonl").log(result, name="baseline")

summary = jupyter_summary(result)
```

`QCAFitResult` から直接呼ぶこともできます。

```python
result.to_markdown_report("reports/qca_result.md")
result.to_latex("reports/solutions.tex", table="solutions")
result.to_jupyter_summary()
```

---

## 可視化

Matplotlib は optional dependency です。

```python
from qca.viz import (
    plot_configuration_chart,
    plot_truth_table,
    plot_xy,
)

plot_truth_table(model)
plot_configuration_chart(result)
plot_xy(model, "policy")
```

---

## 主要機能 7: Machine-Learning-Enhanced QCA

optional packageの `qca.mlqca` は、公開されたmlQCAプロトコルを参考に
Pythonネイティブで実装したワークフローです。XGBoostを条件の重要度評価と
分割閾値の抽出に使用し、その後のcalibration、組合せ探索、QCA評価、
レポート、安定性分析にはPyQCA自身の実装を使用します。

PyQCA組み込みの重要度表はXGBoostのprediction contributionsを使用するため、
通常の利用では外部SHAPパッケージは必須ではありません。

### XGBoostからcsQCA探索まで

```python
import numpy as np
import pandas as pd

from qca import (
    MLQCAConfig,
    fit_xgboost_predictor,
    search_csqca_models,
    validate_mlqca_input,
)

rng = np.random.default_rng(201)
raw_df = pd.DataFrame(
    {
        "policy": rng.normal(size=80),
        "capacity": rng.normal(size=80),
        "support": rng.normal(size=80),
    }
)
raw_df["Y"] = (
    raw_df["policy"] + 0.5 * raw_df["capacity"] > 0
).astype(int)

validated = validate_mlqca_input(raw_df, outcome="Y")
config = MLQCAConfig(
    mode="radical",
    top_k=3,
    model_size=2,
    random_state=201,
    model_params={
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.2,
    },
)

predictor = fit_xgboost_predictor(validated, config)
result = search_csqca_models(validated, predictor, config)

print(predictor.feature_importance)
print(predictor.cutoff_candidates)
print(result.pareto_models)
print(result.best_qca_result)
```

`mode="radical"` は重要度上位条件から組合せを生成します。
`mode="conservative"` では、required/excluded conditionsまたは
`conservative_models` による理論的制約が必要です。

crisp、fuzzy、multi-valueを混在させる場合は
`fit_gsqca_from_predictor()` から `GSQCA` に接続できます。
fuzzy anchor候補は `run_anchor_sensitivity()` で感度分析できます。

### CV / bootstrap安定性

PyQCAでは層化cross-validationと層化bootstrapを利用できます。
cross-validationは各holdout fold、bootstrapは両方のoutcome classを含む
out-of-bag casesで予測性能を評価します。

```python
from pathlib import Path

from qca import bootstrap_mlqca, cross_validate_mlqca

cv_result = cross_validate_mlqca(
    validated,
    config,
    n_splits=5,
    top_k=3,
)
bootstrap_result = bootstrap_mlqca(
    validated,
    config,
    n_bootstrap=100,
    top_k=3,
)

print(cv_result.run_summary)
print(cv_result.feature_stability)
print(bootstrap_result.cutoff_stability)

Path("reports").mkdir(parents=True, exist_ok=True)
cv_result.to_markdown("reports/mlqca_cv.md")
```

安定性表には、上位条件への選択率、treeでの使用率、順位とcontributionの
ばらつき、丸めたcutoffの選択率が含まれます。再現可能なresamplingには
固定した `random_state` を指定してください。

### 論文公開例の再現

テストにはmlQCA公式リポジトリの公開 `voteData` を機械的にCSV変換した
fixtureを収録しています。seed `201` と公開探索範囲内の代表的な設定で、
次を検証しています。

* 公開データ427 casesでの学習
* training accuracyとROC AUCがともに `1.0`
* tutorialで示された6条件のうち5条件以上がPyQCAでも上位10条件以内
* 上位10条件から4条件を選択する210組合せ

これはプロトコル水準の再現テストです。R/caretとPython版XGBoostが
byte単位で同じtreeや順位を生成することを保証するものではありません。
上位10条件の境界付近はXGBoostやPython runtimeのversionによって変動する
可能性があります。fixtureの出典は `tests/data/README.md` と
`THIRD_PARTY_NOTICES.md` に記録しています。

---

## アーキテクチャ概要

PyQCA は、`QCAEngineBase` を共通基底として、`CSQCA`, `MVQCA`, `FSQCA` を first-class engine として実装しています。

`GSQCA` は、それらと同じ共通コアを使う高レベル facade です。crisp / multi-value / fuzzy 条件を同じ分析ワークフローで扱うための入口として設計されています。

```text
qca
├── _constants.py / _types.py / _version.py
├── calibration
│   ├── _validators.py
│   ├── crisp.py
│   ├── logistic.py
│   └── piecewise.py
├── core
│   ├── conditions.py
│   ├── literals.py
│   └── results.py
├── engines
│   ├── _helpers.py
│   ├── base.py
│   ├── csqca.py
│   ├── fsqca.py
│   ├── mvqca.py
│   └── gsqca.py
├── minimizers
│   ├── algorithms.py / engine.py / implicant.py / remainder.py
│   ├── backends.py / benchmark.py
│   ├── standard.py / qmc.py
│   └── set_cover.py
├── mlqca
│   ├── backend.py / config.py / schema.py / validation.py
│   ├── xgboost.py / importance.py / cutoffs.py
│   ├── calibration.py / csqca.py / fuzzy.py / gsqca.py
│   ├── search.py / stability.py / results.py
│   └── reporting.py / reproducibility.py / viz.py
├── reporting
│   ├── _formatters.py
│   ├── export.py / jupyter.py / report.py
│   └── reproducibility.py
├── results
│   └── solution.py
├── sweep
│   ├── anchors.py
│   └── threshold.py
└── viz
    ├── _helpers.py
    ├── bar_plots.py / configuration_chart.py / coverage.py
    └── dashboard.py / distribution.py / truth_table.py / xy_plot.py
```

`qca.core` は条件スキーマ、リテラル、結果dataclassを管理します。engineクラスは `qca.engines` とパッケージトップレベルから公開されます。

先頭が `_` のモジュールはパッケージ内部の共通実装であり、安定した公開APIには含まれません。

---

## 設計原則

### 1. Pythonネイティブ

pandas、NumPy、Jupyter、scikit-learnなどのPythonデータ分析環境で自然に
利用できるAPIを提供します。

### 2. 方法論的な透明性

calibrated data、truth table、solution、consistency、coverageに加え、
mlQCAのfeature importance、split threshold、calibration proposal、
candidate model、Pareto frontier、CV/bootstrap安定性表を確認できます。

QCAとmlQCAをblack boxとして扱わず、予測上の根拠、calibrationの選択、
失敗したmodel評価、resamplingによる変動を追跡可能にします。

### 3. GSQCAをfirst-classに扱う

crisp、multi-value、fuzzy条件の混在を例外扱いせず、`GSQCA` の主要な
利用形態として扱います。

### 4. 最小化アルゴリズムを交換可能にする

standard、QMC、set-covering系backendを同じengineから選択・比較できます。

### 5. 閾値を分析対象として扱う

単一のcalibrationを固定せず、threshold sweepとanchor sensitivityにより
結果が合理的な閾値範囲でどのように変化するかを確認できます。

### 6. Machine learningは根拠であり権威ではない

mlQCAのpredictive importanceは、条件screening、calibration proposal、
model space探索のための経験的根拠として利用します。理論、因果解釈、
QCAの方法論的仮定を置き換えるものではありません。

`radical` modeは重要度上位条件を探索し、`conservative` modeは理論的制約を
要求します。Pareto評価、論文例の再現、CV/bootstrap安定性を公開することで、
提案されたconfigurationが単一のmodel fitへ依存していないか検討できます。

---

## 既存QCAツールとの関係

| Tool / Approach | 主な位置づけ |
| --- | --- |
| R `QCA` package | Rにおける確立されたQCA workflow |
| scpQCA系algorithm | set-coveringを利用する代替的な最小化 |
| ThSQCA | threshold-sweepによる感度分析 |
| R `mlQCA` package | predictive condition selectionとcalibration evidenceを用いる公開mlQCAプロトコル |
| PyQCA | core QCA / GSQCA、最小化、感度分析、独立実装の`qca.mlqca`を統合するPython-native platform |

PyQCAは既存ツールを置き換えるのではなく、相互検証可能なPython実装として
補完することを目指します。`qca.mlqca` はHuangのmlQCA研究を方法論的に
参照していますが、PyQCA独自のdata model、engine、result、reporting、
stability analysisで実装しています。

---

## 開発ステータス

PyQCAは現在experimentalです。現時点では方法論的な正確性、再現性、
明確な公開APIを優先して開発しています。v1.0まではbreaking changeが
発生する可能性があります。

現在の主な利用対象:

* 探索的研究
* prototype analysis
* benchmarking
* 教育用example
* 方法論の検証

査読論文や高リスクな実務判断で利用する場合は、重要な結果を既存QCA
software、方法論上の期待、source dataと照合してください。

今後の開発項目は
[GitHub Issues](https://github.com/t-yamsaki/PyQCA/issues)、公開versionと
release noteは
[GitHub Releases](https://github.com/t-yamsaki/PyQCA/releases)で管理します。

---

## 引用

PyQCAを研究で利用する場合は [`CITATION.cff`](CITATION.cff) の情報を用いて
softwareを引用してください。

`qca.mlqca` を利用する場合は、PyQCAに加えて方法論の参照元である次の論文も
引用してください。

> Huang, Q. (2025). Towards machine-learning enhanced QCA: optimizing
> coverage and empirical significance. *Quality & Quantity*, 59, 4259-4281.
> https://doi.org/10.1007/s11135-025-02146-2

PyQCAの引用はsoftware実装を、Huang論文の引用はoptional mlQCA workflowへ
影響を与えた方法論を示します。

---

## コントリビューション

コントリビューションはプロジェクトのMIT Licenseで受け入れます。開発手順と
出典要件は [`CONTRIBUTING.md`](CONTRIBUTING.md) を参照してください。

`qca.mlqca` への貢献では、方法論上の参照元、datasetのprovenance、依存packageの
license、独立実装かadaptationかを明記してください。GPL-licensed R sourceを
MIT-licensed codebaseへ翻訳・複製しないでください。再現用fixtureには再配布可能な
licenseが必要です。

---

## ライセンス

MIT License

プロジェクトのライセンスは [`LICENSE`](LICENSE)、第三者への帰属は
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)、コード由来監査は
[`PROVENANCE_AUDIT.md`](PROVENANCE_AUDIT.md) を参照してください。

研究利用時の引用情報は [`CITATION.cff`](CITATION.cff)、コントリビューションの
ライセンス条件は [`CONTRIBUTING.md`](CONTRIBUTING.md) に記載しています。

PRIは配置・結果・結果の否定の三集合の共通部分を使う標準式で計算します。`pri_cut` による行選択を修正しました。詳細は [PRIの仕様](docs/user_guide/core_qca.rst) を参照してください。

既定の `standard` と `qmc` は厳密な被覆選択を行い、項数、総論理複雑度の順に最小化します。最適解を1つ返し、同等の最適解は全列挙しません。探索時間は指数的に増える可能性があり、近似法への自動切り替えはありません。近似を許容する場合は `greedy_set_cover` を明示してください。詳細は[最小化の仕様](docs/user_guide/minimization.rst)を参照してください。

十分性の適合度は、全事例の帰属度が0.5以下でも正の帰属度を用いて計算します。事例数による候補除外は別に適用します。詳細は[低帰属度の十分性評価](docs/user_guide/core_qca.rst)を参照してください。

ロジスティック校正はクロスオーバーの両側で傾きを分け、非対称な3アンカーを満たします。カスタム帰属度は0.5を挟む必要があり、非対称な曲線では単一の傾き・切片の代わりに区間別係数を返します。詳細は[校正の仕様](docs/user_guide/calibration.rst)を参照してください。

集合条件の帰属度が正確に0.5の事例は、警告とともに真理表の行割当・頻度から除外します。整合度とPRIには元の全事例を使います。詳細は[クロスオーバーの仕様](docs/user_guide/core_qca.rst)を参照してください。

多値条件のメンバーシップは、真理表の行割当と同じ条件スキーマで解決します。カテゴリ条件では、`A=0`、`A[0]`、`A__0`という補助列が存在してもカテゴリ列を使用します。GSQCAでは、`value_columns`を明示した場合、値別列と`calibrated=True`を指定した場合、またはカテゴリ列がなく値別列のみの場合に値別メンバーシップを使用します。単独の`MultiValueLiteral.membership(data)` APIにはエンジンのスキーマがないため、従来の値別列優先の動作を維持します。

解の式と被覆表のラベルでは、多値カテゴリを`A=0`、`A=1`、`A=2`のように明示します。クリスプ／ファジィ集合条件では`A`・`~A`表記を維持します。低水準APIの`Implicant.label`と`build_coverage_table`には任意の`is_multivalue`フラグを指定できます。型情報を省略した呼び出しでは、整数0・1の従来の集合表記を維持します。

必要条件の自明性は、高いcoverageではなく「必要性を満たし、全事例の条件帰属度が正確に1」の場合に判定します。低いcoverageだけでも自明とは判定しません。手動生成の必要条件結果で任意の`is_universal`情報がない場合、`is_trivial=None`（判定不能）を返します。詳細と互換性は[自明性の仕様](docs/user_guide/core_qca.rst)を参照してください。

真理表の`include`は、整合度・PRIに加えて`frequency_cutoff`を反映します。`fit`の返却結果と単独の`build_truth_table`で利用できます。低頻度行は表示を残し、最小化では従来どおり論理残余として扱います。詳細は[頻度と採択表示の仕様](docs/user_guide/core_qca.rst)を参照してください。

必要性の整合度・被覆度は、`1e-12`以下でも正の帰属度合計を定義式どおり計算し、分母が正確に0の場合のみ0を返します。ケース数の判定は変更しません。詳細は[低帰属度の必要性評価](docs/user_guide/core_qca.rst)を参照してください。

## 0.3.0の仕様と互換性

- `standard`、`qmc`、`set_cover`、`exact_set_cover` は項数、論理的複雑度の順に最適な被覆を1つ選択します。全最適解の列挙や自動的な近似への切り替えは行いません。近似には `greedy_set_cover` を明示します。
- 中間解は正の観測配置から支持される容易な反実仮想を使用し、選択された簡約解の領域内に限定します。複雑解 ⊆ 中間解 ⊆ 簡約解の関係を保ちます。
- PRIの三重共通部分、微小な正の帰属度の必要性・十分性指標、0.5の条件値の行割当除外、頻度基準の `include` への反映を修正しています。
- 校正では非対称ロジスティックアンカー、微小間隔、重複インデックスを扱い、piecewiseの上下限が0.5を挟むことを検証します。
- 閾値感度分析では欠損・非有限値を含む実行をエラーとして記録し、感度分析で出力名の衝突を拒否します。
- mlQCAではXGBoostの特徴名をモデル情報に照合します。SHAPの方向推定は単位変更に対して安定な相関を使い、方向不明時は明示指定を要求します。校正方向の別名・大文字・空白も正規化して計算と返却メタデータに反映します。
- mlQCAアンカー感度分析の方向指定は元の特徴名と校正後の名前を受理します。どちらも同じ校正集合への所属を意味し、low方向でも指定値を反転しません。

詳しい定義と移行上の変更は [Sphinx変更履歴](https://pyqca.readthedocs.io/en/latest/release_notes.html)と各ユーザーガイドを参照してください。
