import warnings, os, json, time
os.environ["PYTHONWARNINGS"] = "ignore"  # propagate suppression into joblib subprocess workers
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
CLASS_NAMES = ["Normal", "Suspect", "Pathological"]
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)
sns.set_style("whitegrid")

# FAST_MODE trims the resampler/model grid and CV folds for a quick Colab run.
# Set False to restore the full original grid (5 resamplers x 6 models x 5-fold CV).
FAST_MODE = True
N_SPLITS = 3 if FAST_MODE else 5


# ============================================================================
# 1. LOAD
# ============================================================================
def load_data(path="fetal_health.csv"):
    df = pd.read_csv(path)
    print(f"[LOAD] {df.shape[0]} rows x {df.shape[1]} cols | "
          f"missing={df.isnull().sum().sum()} | duplicates={df.duplicated().sum()}")
    return df


# ============================================================================
# 2. EDA
# ============================================================================
def run_eda(df):
    counts = df["fetal_health"].value_counts().sort_index()
    print("[EDA] Class distribution:")
    for cls, name in zip([1.0, 2.0, 3.0], CLASS_NAMES):
        n = int(counts.get(cls, 0))
        print(f"       {name:<13}: {n:>5} ({n/len(df)*100:5.1f}%)")
    ir = counts.max() / counts.min()
    print(f"[EDA] Imbalance ratio (max/min) = {ir:.2f}")

    plt.figure(figsize=(6, 4))
    sns.barplot(x=CLASS_NAMES, y=counts.values, palette="viridis")
    plt.title("Class Distribution"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig1_class_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(14, 11))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig2_correlation.png", dpi=150); plt.close()


# ============================================================================
# 3. PREPROCESS
# ============================================================================
def preprocess(df):
    X = df.drop(columns=["fetal_health"]).values
    feat_names = df.drop(columns=["fetal_health"]).columns.tolist()
    y = (df["fetal_health"].astype(int) - 1).values  # 0,1,2

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, feat_names, scaler


# ============================================================================
# 4-5. MODEL & RESAMPLER FACTORIES
# ============================================================================
def get_resamplers():
    full = {
        "None": None,
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=RANDOM_STATE),
        "ADASYN": ADASYN(random_state=RANDOM_STATE),
        "SMOTETomek": SMOTETomek(random_state=RANDOM_STATE),
    }
    if FAST_MODE:
        return {k: full[k] for k in ["None", "SMOTE", "SMOTETomek"]}
    return full


def get_models(final=False):
    """final=True is used only once, for the single selected model after
    screening, so it's safe to enable the costlier SVM probability calibration
    there without paying that cost across the whole benchmark grid."""
    full = {
        "LogReg": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        # n_jobs=1 on individual models: the outer CV loop already parallelizes
        # across folds with n_jobs=-1, so parallelizing here too would
        # oversubscribe Colab's 2 vCPUs and slow everything down.
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                                 eval_metric="mlogloss", random_state=RANDOM_STATE,
                                 verbosity=0, n_jobs=1),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE,
                                   verbose=-1, n_jobs=1),
        # probability=True triggers an internal 5-fold Platt calibration per fit;
        # only turn it on for the final, once-only refit.
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=final, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True,
                             n_iter_no_change=10, random_state=RANDOM_STATE),
    }
    if FAST_MODE:
        return {k: full[k] for k in ["LogReg", "RandomForest", "XGBoost", "LightGBM"]}
    return full


# ============================================================================
# 6. CROSS-VALIDATED RESAMPLING x MODEL BENCHMARK
# ============================================================================
def scorer_suite(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MacroF1": f1_score(y_true, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "PathologicalRecall": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0)["2"]["recall"],
    }


def benchmark(X_tr, y_tr):
    """Stratified N_SPLITS-fold CV over every (resampler x model) combination."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    print(f"\n[BENCHMARK] resampler x model, {N_SPLITS}-fold CV...")
    for r_name, resampler in get_resamplers().items():
        for m_name, model in get_models().items():
            steps = []
            if resampler is not None:
                steps.append(("resample", resampler))
            steps.append(("clf", model))
            pipe = ImbPipeline(steps)
            # resampling applied inside each fold only (no leakage)
            y_cv = cross_val_predict(pipe, X_tr, y_tr, cv=cv, n_jobs=-1)
            metrics = scorer_suite(y_tr, y_cv)
            metrics.update({"Resampler": r_name, "Model": m_name})
            rows.append(metrics)
            print(f"       {r_name:<16} {m_name:<13} "
                  f"MacroF1={metrics['MacroF1']:.4f} MCC={metrics['MCC']:.4f} "
                  f"PathRec={metrics['PathologicalRecall']:.4f}")
    res = pd.DataFrame(rows)[
        ["Resampler", "Model", "Accuracy", "BalancedAcc",
         "MacroF1", "MCC", "Kappa", "PathologicalRecall"]
    ].sort_values(["MCC", "MacroF1"], ascending=False).reset_index(drop=True)
    res.to_csv(f"{OUTDIR}/table1_benchmark.csv", index=False)
    print(f"\n[BENCHMARK] Top 5 (by MCC):\n{res.head(5).to_string(index=False)}")
    return res


# ============================================================================
# 7. TUNE BEST COMBINATION
# ============================================================================
def tune_best(best_model_name, best_resampler_name, X_tr, y_tr):
    grids = {
        "XGBoost": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.1, 0.2],
            "clf__subsample": [0.8, 1.0],
        },
        "LightGBM": {
            "clf__n_estimators": [200, 400, 600],
            "clf__num_leaves": [31, 63, 127],
            "clf__learning_rate": [0.03, 0.1, 0.2],
        },
        "RandomForest": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        },
    }
    grid = grids.get(best_model_name, grids["XGBoost"])
    resampler = get_resamplers()[best_resampler_name]
    model = get_models(final=True)[best_model_name]
    steps = ([("resample", resampler)] if resampler else []) + [("clf", model)]
    pipe = ImbPipeline(steps)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    n_iter = 10 if FAST_MODE else 15
    print(f"\n[TUNE] RandomizedSearchCV ({n_iter} iters) on {best_resampler_name} + {best_model_name}...")
    search = RandomizedSearchCV(
        pipe, grid, n_iter=n_iter, scoring="f1_macro",
        cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    search.fit(X_tr, y_tr)
    print(f"[TUNE] Best Macro-F1 (CV) = {search.best_score_:.4f}")
    print(f"[TUNE] Best params = {json.dumps(search.best_params_)}")
    return search.best_estimator_


# ============================================================================
# 8-9. FINAL EVAL + ROC/PR CURVES
# ============================================================================
def final_eval(model, X_te, y_te, tag="best_model"):
    y_pred = model.predict(X_te)
    m = scorer_suite(y_te, y_pred)
    try:
        y_proba = model.predict_proba(X_te)
        m["ROC_AUC_ovr"] = roc_auc_score(
            label_binarize(y_te, classes=[0, 1, 2]), y_proba,
            average="macro", multi_class="ovr")
    except Exception:
        y_proba = None
        m["ROC_AUC_ovr"] = np.nan

    print(f"\n[FINAL] Held-out test metrics:")
    for k, v in m.items():
        print(f"        {k:<20}: {v:.4f}")
    print("\n" + classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (best model)"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig3_confusion_matrix.png", dpi=150); plt.close()

    if y_proba is not None:
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        plt.figure(figsize=(6, 5))
        for i, name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (One-vs-Rest)"); plt.legend()
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig4_roc_curves.png", dpi=150); plt.close()

        plt.figure(figsize=(6, 5))
        for i, name in enumerate(CLASS_NAMES):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            plt.plot(rec, prec, label=f"{name} (AP-AUC={auc(rec, prec):.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (One-vs-Rest)"); plt.legend()
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/fig5_pr_curves.png", dpi=150); plt.close()

    pd.DataFrame([m]).to_csv(f"{OUTDIR}/table2_final_metrics.csv", index=False)
    return m


# ============================================================================
# 10. SHAP EXPLAINABILITY (XAI)
# ============================================================================
def explain(model, X_tr, X_te, feat_names):
    try:
        import shap
    except ImportError:
        print("[SHAP] shap not installed; skipping explainability.")
        return
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    name = type(clf).__name__
    print(f"\n[SHAP] Explaining {name} with TreeExplainer...")
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_te)
        plt.figure()
        shap.summary_plot(sv, X_te, feature_names=feat_names,
                          class_names=CLASS_NAMES, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fig6_shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[SHAP] Saved fig6_shap_importance.png")
    except Exception as e:
        print(f"[SHAP] Skipped (model may not be tree-based): {e}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    df = load_data()
    run_eda(df)
    X_tr, X_te, y_tr, y_te, feat_names, scaler = preprocess(df)

    bench = benchmark(X_tr, y_tr)
    best = bench.iloc[0]
    print(f"\n[SELECT] Winner: {best['Resampler']} + {best['Model']} "
          f"(MCC={best['MCC']:.4f}, MacroF1={best['MacroF1']:.4f})")

    best_model = tune_best(best["Model"], best["Resampler"], X_tr, y_tr)
    final_eval(best_model, X_te, y_te)
    explain(best_model, X_tr, X_te, feat_names)

    joblib.dump({"model": best_model, "scaler": scaler, "features": feat_names},
                f"{OUTDIR}/best_model.joblib")
    print(f"\n[DONE] All tables + figures + best_model.joblib saved in '{OUTDIR}/' "
          f"in {time.time()-t0:.1f}s (FAST_MODE={FAST_MODE})")


if __name__ == "__main__":
    main()
