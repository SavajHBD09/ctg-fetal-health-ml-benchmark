<div align="center">

<br/>

```
 ██████╗████████╗ ██████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔════╝╚══██╔══╝██╔════╝     ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██║        ██║   ██║  ███╗    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██║        ██║   ██║   ██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
╚██████╗   ██║   ╚██████╔╝    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
 ╚═════╝   ╚═╝    ╚═════╝     ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
```

### _No leakage. No inflated accuracy. No black boxes._

> A leakage-free, imbalance-aware machine learning benchmark that classifies fetal health
> from cardiotocography data — and explains every prediction with SHAP.

<br/>

[![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge&logo=leaflet&logoColor=white)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC3C3C?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-6C5CE7?style=for-the-badge&logo=databricks&logoColor=white)](https://shap.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

<br/>

[![License](https://img.shields.io/badge/License-Educational-FF6B6B?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Major_Project--I_2026-4ADE80?style=flat-square)]()
[![Reproducible](https://img.shields.io/badge/Reproducible-seed_42-818CF8?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-94.6%25-06B6D4?style=flat-square)]()
[![MCC](https://img.shields.io/badge/MCC-0.850-14B8A6?style=flat-square)]()

</div>

---

## 👥 Team

<div align="center">

| #   | Name                 |   Role                        |
| --- | -------------------- |  --------------------------- |
| 🧠  | **Harshit Dave**     | ML Engineer & Researcher    |
| 🎓  | **Prof. Dhara Joshi** |  Internal Guide              |

_Department of Computer Engineering · Faculty of Engineering & Technology · Marwadi University, Rajkot_

</div>

---

## 📑 Table of Contents

- [🩺 The Problem](#-the-problem)
- [✨ What This Project Does](#-what-this-project-does)
- [📊 Results](#-results)
- [🛠 Tech Stack](#-tech-stack)
- [🏗 Pipeline Architecture](#-pipeline-architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Configuration](#️-configuration)
- [📈 Outputs](#-outputs)
- [🧠 Methodology](#-methodology)
- [📚 Dataset](#-dataset)
- [📄 Citation](#-citation)
- [📝 License](#-license)

---

## 🩺 The Problem

Cardiotocography (CTG) is the standard non-invasive way to monitor fetal heart rate and uterine
contractions during pregnancy. But reading a CTG trace is a **subjective** skill — two clinicians
looking at the same trace often disagree, and the same clinician may disagree with themselves
on a different day.

So: automate it. Except there's a catch.

<table>
<tr>
<td width="50%" valign="top">

### ⚠️ The Imbalance Trap

Only **8.3%** of records are Pathological — the cases that actually need intervention.

```
Normal        ████████████████████  77.8%
Suspect       ███                   13.9%
Pathological  ██                     8.3%
                       imbalance ratio → 9.4 : 1
```

A model that **always predicts Normal** scores 77.8% accuracy — while detecting
exactly **zero** distressed fetuses.

</td>
<td width="50%" valign="top">

### 🕳️ The Leakage Trap

Most published CTG pipelines scale and oversample **before** splitting the data.

```python
# ❌ WRONG — test data leaks into training
X = scaler.fit_transform(X)
X, y = SMOTE().fit_resample(X, y)
X_tr, X_te = train_test_split(X, y)
```

Synthetic minority samples end up in the test set. Reported scores look great
and collapse in deployment.

</td>
</tr>
</table>

---

## ✨ What This Project Does

<table>
<tr>
<td width="50%" valign="top">

### 🔒 Leakage-Free By Construction

Scaling and resampling live **inside** `ImbPipeline`, so they only ever see
training folds. Leakage isn't avoided by discipline — it's structurally impossible.

### 🧪 Systematic Benchmark

Every resampler × every classifier, cross-validated. No cherry-picking a
single favourite configuration.

### 📐 Imbalance-Aware Selection

Ranked by **MCC**, not accuracy. Balanced Accuracy, Macro-F1, Cohen's Kappa,
and Pathological recall reported alongside.

</td>
<td width="50%" valign="top">

### 🎯 Single Held-Out Evaluation

The tuned model touches the test set **exactly once** — no repeated peeking,
no selection bias.

### 🔍 SHAP Explainability

Every prediction comes with feature attributions in clinical language, so a
doctor can check the model's reasoning.

### 🎲 Fully Reproducible

`random_state = 42` across splitting, CV, resampling, and model init.
Same input → same output, every time.

</td>
</tr>
</table>

---

## 📊 Results

### 🏆 Winning Configuration — `None + LightGBM`

<div align="center">

| Metric                    | Score      | What it means                                 |
| ------------------------- | ---------- | --------------------------------------------- |
| 🎯 **Accuracy**           | **94.6%**  | Overall correct classifications               |
| 🧮 **MCC**                | **0.850**  | Imbalance-robust agreement _(primary metric)_ |
| ⚖️ **Balanced Accuracy**  | **0.886**  | Mean per-class recall                         |
| 📊 **Macro-F1**           | **0.893**  | Equal weight to every class                   |
| 🤝 **Cohen's Kappa**      | **0.849**  | Agreement beyond chance                       |
| 🚨 **Pathological Recall**| **91.4%**  | **32 of 35** sick fetuses caught              |
| 📈 **ROC-AUC** _(OvR)_    | **0.985**  | Discriminative power                          |

</div>

### 🔢 Confusion Matrix

```
                    ┌─────────── PREDICTED ───────────┐
                      Normal    Suspect   Pathological
        ┌ Normal    │   326   │     5   │      1     │   98.2% recall
 TRUE   │ Suspect   │    10   │    45   │      4     │   76.3% recall
        └ Patholog. │     2   │     1   │     32     │   91.4% recall ✅
```

### 📋 Full Benchmark — All 12 Configurations

<details>
<summary><strong>Click to expand the complete ranking (by MCC)</strong></summary>

<br/>

| Rank | Resampler   | Model               | Accuracy | Bal. Acc. | Macro-F1 | MCC       | Path. Recall |
| ---- | ----------- | ------------------- | -------- | --------- | -------- | --------- | ------------ |
| 🥇 1 | None        | **LightGBM**        | 0.962    | 0.916     | 0.928    | **0.894** | 0.936        |
| 🥈 2 | None        | XGBoost             | 0.958    | 0.913     | 0.926    | 0.883     | 0.936        |
| 🥉 3 | SMOTE-Tomek | LightGBM            | 0.954    | 0.916     | 0.918    | 0.875     | 0.936        |
| 4    | SMOTE       | XGBoost             | 0.953    | 0.913     | 0.915    | 0.872     | 0.929        |
| 5    | SMOTE       | LightGBM            | 0.953    | 0.912     | 0.916    | 0.871     | 0.929        |
| 6    | SMOTE-Tomek | XGBoost             | 0.952    | 0.914     | 0.914    | 0.870     | 0.929        |
| 7    | SMOTE-Tomek | Random Forest       | 0.948    | 0.917     | 0.909    | 0.861     | **0.943** ⭐ |
| 8    | SMOTE       | Random Forest       | 0.947    | 0.915     | 0.907    | 0.858     | **0.943** ⭐ |
| 9    | None        | Random Forest       | 0.945    | 0.880     | 0.899    | 0.846     | 0.894        |
| 10   | None        | Logistic Regression | 0.897    | 0.795     | 0.804    | 0.716     | 0.787        |
| 11   | SMOTE       | Logistic Regression | 0.874    | 0.849     | 0.789    | 0.711     | 0.844        |
| 12   | SMOTE-Tomek | Logistic Regression | 0.874    | 0.849     | 0.789    | 0.711     | 0.844        |

⭐ _Highest Pathological recall in the grid — worth choosing if catching every sick case matters more than aggregate MCC._

</details>

### 🔍 What the Model Actually Learned

SHAP ranked these as the top predictors — and every one is an **established obstetric indicator**:

```
1. abnormal_short_term_variability     ████████████████████  ← classic distress marker
2. accelerations                       ███████████████████   ← reassuring sign of wellbeing
3. % time abnormal long_term_variability ████████████        ← key Suspect driver
4. histogram_mean                      ███████████           ← FHR distribution shape
5. uterine_contractions                ███████
```

> 💡 The model isn't exploiting a statistical artefact — it converged on the same signals
> clinicians have used for decades. That's what makes it trustworthy.

---

## 🛠 Tech Stack

### 🧬 Core ML

| Tool                                                       | Version  | Purpose                            |
| ---------------------------------------------------------- | -------- | ---------------------------------- |
| [Python](https://python.org)                               | `3.12`   | Language                           |
| [scikit-learn](https://scikit-learn.org)                   | latest   | Models, CV, metrics, scaling       |
| [LightGBM](https://lightgbm.readthedocs.io)                | latest   | 🏆 Winning classifier              |
| [XGBoost](https://xgboost.readthedocs.io)                  | latest   | Gradient boosting comparison       |
| [imbalanced-learn](https://imbalanced-learn.org)           | latest   | SMOTE, SMOTE-Tomek, leak-safe pipe |

### 📊 Analysis & Explainability

| Tool                                        | Purpose                              |
| ------------------------------------------- | ------------------------------------ |
| [SHAP](https://shap.readthedocs.io)         | Shapley-value feature attribution    |
| [Pandas](https://pandas.pydata.org)         | Data loading & manipulation          |
| [NumPy](https://numpy.org)                  | Numerical operations                 |
| [Matplotlib](https://matplotlib.org)        | Figure generation                    |
| [Seaborn](https://seaborn.pydata.org)       | Statistical plots & heatmaps         |
| [Joblib](https://joblib.readthedocs.io)     | Model persistence                    |

---

## 🏗 Pipeline Architecture

```
┌──────────────┐   ┌──────────────┐   ┌─────────────────────────────────────┐
│  CTG Dataset │──▶│  Stratified  │──▶│         🔒 LEAKAGE BARRIER          │
│  2,126 × 21  │   │  80/20 Split │   │  nothing below sees test data       │
└──────────────┘   └──────────────┘   └──────────────┬──────────────────────┘
                                                     │
                   ┌─────────────────────────────────▼──────────────────────┐
                   │  StandardScaler   ·  fit on TRAIN only                 │
                   │  Resampler        ·  fit inside each CV FOLD only      │
                   └─────────────────────────────────┬──────────────────────┘
                                                     │
                   ┌─────────────────────────────────▼──────────────────────┐
                   │  Benchmark  →  3 resamplers × 4 models × 5-fold CV     │
                   │  Rank by MCC  →  Tune winner (RandomizedSearchCV)      │
                   └─────────────────────────────────┬──────────────────────┘
                                                     │
        ┌────────────────────────────────────────────▼────────────────────┐
        │   🎯 Held-out test · evaluated ONCE      🔍 SHAP explanations   │
        └─────────────────────────────────────────────────────────────────┘
```

| Stage | Component            | Responsibility                                        |
| ----- | -------------------- | ----------------------------------------------------- |
| 1     | **Ingestion**        | Load CTG data, verify completeness & duplicates        |
| 2     | **Partitioning**     | Stratified 80/20 split preserving class ratios         |
| 3     | **Scaling**          | `StandardScaler` fitted on training partition only     |
| 4     | **Resampling**       | Synthetic samples generated inside training folds only |
| 5     | **Benchmarking**     | Every resampler × classifier, stratified 5-fold CV     |
| 6     | **Selection**        | Rank by MCC, tune with `RandomizedSearchCV`            |
| 7     | **Final Evaluation** | One-shot test on untouched hold-out set                |
| 8     | **Explainability**   | `TreeExplainer` → per-class SHAP attributions          |

---

## 📁 Project Structure

```
ctg-fetal-health-ml-benchmark/
├── 🐍 fetal_health_pipeline.py     # Main pipeline — run this
├── 📊 fetal_health.csv             # Dataset (2,126 × 22)
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md
│
├── 📂 results/                     # Auto-generated on run
│   ├── 🖼️  fig1_class_distribution.png
│   ├── 🖼️  fig2_correlation.png
│   ├── 🖼️  fig3_confusion_matrix.png
│   ├── 🖼️  fig4_roc_curves.png
│   ├── 🖼️  fig5_pr_curves.png
│   ├── 🖼️  fig6_shap_importance.png
│   ├── 📈 table1_benchmark.csv     # All configurations ranked
│   ├── 📈 table2_final_metrics.csv # Held-out test metrics
│   └── 🤖 best_model.joblib        # Trained model + scaler
│
└── 📂 docs/
    ├── 📄 methodology.md           # Detailed method write-up
    └── 📄 paper-draft.pdf          # Research manuscript
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Download                         |
| ----------- | ------- | -------------------------------- |
| Python      | `3.10+` | [python.org](https://python.org) |
| pip         | latest  | ships with Python                |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/SavajHBD09/ctg-fetal-health-ml-benchmark.git
cd ctg-fetal-health-ml-benchmark
```

---

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Or install manually</summary>

<br/>

```bash
pip install pandas numpy scikit-learn imbalanced-learn \
            xgboost lightgbm shap joblib matplotlib seaborn
```

</details>

---

### Step 3 — Run the Pipeline

```bash
python fetal_health_pipeline.py
```

> ✅ Every figure, table, and the trained model land in **`results/`**

---

### ☁️ Run on Google Colab

<details>
<summary><strong>Three cells, no local setup</strong></summary>

<br/>

**Cell 1 — install**

```python
!pip install -q imbalanced-learn xgboost lightgbm shap joblib
```

**Cell 2 — fetch the dataset**

```python
import kagglehub, os
path = kagglehub.dataset_download("andrewmvd/fetal-health-classification")
csv_path = os.path.join(path, "fetal_health.csv")
print("Dataset at:", csv_path)
```

**Cell 3 — run**

```python
!python fetal_health_pipeline.py
```

**Download your results**

```python
!zip -r results.zip results
from google.colab import files
files.download("results.zip")
```

</details>

---

## ⚙️ Configuration

Open `fetal_health_pipeline.py` and edit the config block near the top:

```python
FAST_MODE = True     # ⚡ quick check  — 3 resamplers × 4 models, 3-fold CV
                     # 🔬 full run     — set to False for 5-fold CV
RANDOM_STATE = 42    # 🎲 controls every random operation
N_SPLITS = 5         # 📊 cross-validation folds
```

| Mode                 | Grid                       | CV     | Runtime  | Use for            |
| -------------------- | -------------------------- | ------ | -------- | ------------------ |
| `FAST_MODE = True`   | 3 resamplers × 4 models    | 3-fold | ~30 sec  | Quick iteration    |
| `FAST_MODE = False`  | 3 resamplers × 4 models    | 5-fold | ~2 min   | 📄 Paper / report  |

> ⚠️ Use **`FAST_MODE = False`** for any results you plan to publish or submit.

---

## 📈 Outputs

Every run regenerates the full result set:

| File                            | Type  | Contents                                        |
| ------------------------------- | ----- | ----------------------------------------------- |
| `fig1_class_distribution.png`   | 🖼️    | Class imbalance bar chart                       |
| `fig2_correlation.png`          | 🖼️    | 21-feature correlation heatmap                  |
| `fig3_confusion_matrix.png`     | 🖼️    | Per-class prediction breakdown                  |
| `fig4_roc_curves.png`           | 🖼️    | One-vs-rest ROC with per-class AUC              |
| `fig5_pr_curves.png`            | 🖼️    | Precision-recall curves _(better for imbalance)_ |
| `fig6_shap_importance.png`      | 🖼️    | Global SHAP feature importance by class         |
| `table1_benchmark.csv`          | 📈    | Every configuration, ranked by MCC              |
| `table2_final_metrics.csv`      | 📈    | Held-out test metrics                           |
| `best_model.joblib`             | 🤖    | Trained model + scaler + feature names          |

<details>
<summary><strong>🔮 Loading the saved model for inference</strong></summary>

<br/>

```python
import joblib, pandas as pd

bundle = joblib.load("results/best_model.joblib")
model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]

# new CTG reading — 21 features in the original order
X_new = pd.DataFrame([[...]], columns=features)
X_scaled = pd.DataFrame(scaler.transform(X_new), columns=features)

pred  = model.predict(X_scaled)[0]          # 0=Normal, 1=Suspect, 2=Pathological
proba = model.predict_proba(X_scaled)[0]

print(["Normal", "Suspect", "Pathological"][pred], proba)
```

</details>

---

## 🧠 Methodology

### 🔒 The Leakage Barrier — the heart of this project

```python
# ✅ RIGHT — resampling confined to training folds
steps = [("resample", resampler), ("clf", model)]
pipe  = ImbPipeline(steps)
y_cv  = cross_val_predict(pipe, X_tr, y_tr, cv=cv, n_jobs=-1)
```

`cross_val_predict` calls `pipe.fit()` on the k−1 training folds and `pipe.predict()` on the
validation fold. `ImbPipeline` applies resampling **only during fit** — so synthetic samples
can never reach validation data. A standard `sklearn.Pipeline` cannot do this.

### 📐 Why MCC Instead of Accuracy

<div align="center">

| Metric               | Imbalance-safe? | Why                                                 |
| -------------------- | :-------------: | --------------------------------------------------- |
| Accuracy             | ❌              | Rewards predicting the 77.8% majority class          |
| **MCC**              | ✅              | Uses all confusion-matrix cells; robust to skew      |
| **Balanced Accuracy**| ✅              | Mean per-class recall — every class weighted equally |
| **Macro-F1**         | ✅              | Unweighted mean of per-class F1                      |
| **Pathological Recall** | ✅ ⭐        | The metric with real clinical cost attached          |

</div>

### 🔄 Resampling Strategies Compared

| Strategy        | Mechanism                                         | Effect                        |
| --------------- | ------------------------------------------------- | ----------------------------- |
| **None**        | Train on the original distribution                | 🏆 Best MCC here              |
| **SMOTE**       | Interpolate between minority neighbours           | Lifts minority recall         |
| **SMOTE-Tomek** | SMOTE + Tomek-link cleanup of overlapping samples | Cleaner class boundaries      |

> 💡 **Counter-intuitive finding:** no resampling won. LightGBM's internal loss handling already
> copes with the imbalance — adding synthetic samples didn't help aggregate MCC. This is exactly
> why benchmarking beats assuming SMOTE is always the answer.

---

## 📚 Dataset

<div align="center">

| Property           | Value                                                     |
| ------------------ | --------------------------------------------------------- |
| 📊 **Records**     | 2,126                                                     |
| 🔢 **Features**    | 21 numerical                                              |
| 🎯 **Classes**     | 3 — Normal / Suspect / Pathological                       |
| ⚖️ **Imbalance**   | 9.4 : 1                                                   |
| 🕳️ **Missing**     | 0                                                         |
| 📥 **Source**      | [UCI](https://archive.ics.uci.edu/dataset/193/cardiotocography) · [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) |

</div>

<details>
<summary><strong>📋 Feature reference</strong></summary>

<br/>

| Feature                                                  | Clinical meaning                          |
| -------------------------------------------------------- | ----------------------------------------- |
| `baseline value`                                         | Average FHR (beats per minute)            |
| `accelerations`                                          | Rate of FHR accelerations per second      |
| `fetal_movement`                                         | Rate of fetal movements per second        |
| `uterine_contractions`                                   | Rate of uterine contractions per second   |
| `light_decelerations`                                    | Rate of light FHR decelerations           |
| `severe_decelerations`                                   | Rate of severe FHR decelerations          |
| `prolongued_decelerations`                               | Rate of prolonged decelerations           |
| `abnormal_short_term_variability`                        | % time with abnormal STV ⭐               |
| `mean_value_of_short_term_variability`                   | Mean STV in milliseconds                  |
| `percentage_of_time_with_abnormal_long_term_variability` | % time with abnormal LTV                  |
| `mean_value_of_long_term_variability`                    | Mean LTV                                  |
| `histogram_width` · `_min` · `_max`                      | FHR histogram spread                      |
| `histogram_number_of_peaks` · `_zeroes`                  | FHR histogram shape                       |
| `histogram_mode` · `_mean` · `_median`                   | FHR histogram central tendency            |
| `histogram_variance` · `_tendency`                       | FHR histogram dispersion & asymmetry      |

⭐ = top SHAP predictor

</details>

> 📖 Original data: Ayres-de-Campos et al. (2000), _SisPorto 2.0: A Program for Automated Analysis
> of Cardiotocograms_, J. Matern. Fetal Med. 9(5), 311–318.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{dave2026ctg,
  author       = {Dave, Harshit Bhardwajbhai and Joshi, Dhara},
  title        = {Predicting Fetal Health Using AI: A Leakage-Free, Imbalance-Aware
                  Benchmark of Machine Learning Models on Cardiotocography Data
                  with SHAP-Based Explainability},
  year         = {2026},
  institution  = {Marwadi University, Rajkot},
  howpublished = {\url{https://github.com/SavajHBD09/ctg-fetal-health-ml-benchmark}}
}
```

---

## ⚠️ Disclaimer

> This is a **research project**, not a medical device. It has been validated on a single public
> dataset and has **not** undergone external clinical validation or regulatory approval.
> It must not be used for actual clinical decision-making.

---

## 📝 License

Built for educational and research purposes as part of Major Project-I at Marwadi University.
Feel free to explore, fork, and build upon it.

---

<div align="center">

## 🙌 Built By

|                  |
| :--------------: |
| **Harshit Dave** |
|  `92410103124`   |

_Guided by **Prof. Dhara Joshi** · Department of Computer Engineering_

<br/>

_"The best metric is the one that punishes you for the mistakes that actually matter."_

<br/>

Made with ❤️ and a healthy distrust of accuracy scores · Major Project-I · 2026
&nbsp;|&nbsp; [🐛 Report Bug](../../issues)
&nbsp;|&nbsp; [💡 Request Feature](../../issues)

<br/>

```
 ______  ______  ______     ______  ______  ______  ______  __
/\  ___\/\__  _\/\  ___\   /\  ___\/\  ___\/\__  _\/\  __ \/\ \
\ \ \___\/_/\ \/\ \ \__ \  \ \  __\\ \  __\\/_/\ \/\ \  __ \ \ \____
 \ \_____\ \ \_\ \ \_____\  \ \_\   \ \_____\ \ \_\ \ \_\ \_\ \_____\
  \/_____/  \/_/  \/_____/   \/_/    \/_____/  \/_/  \/_/\/_/\/_____/
```

</div>
