import os
import warnings
warnings.filterwarnings("ignore")
import joblib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", "outputs", ".mplconfig"))

import pandas as pd
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASETS = {
    "Original":  os.path.join(ROOT, "original_dataset", "final1.csv"),
    "CTGAN":     os.path.join(ROOT, "ctgan", "ctgan_balanced_data.csv"),
    "Adv-CTGAN": os.path.join(ROOT, "adversarial_training", "adv_balanced_data.csv"),
}

CLASSIFIERS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                          random_state=42, early_stopping=True),
}

OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    if "title" in df.columns:
        df = df.drop(columns=["title"])
    if "body" in df.columns:
        df = df.drop(columns=["body"])
    if "is_fraud" in df.columns:
        df = df[df["is_fraud"] != -1].copy()
    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def align_features(X_train, X_test):
    missing = set(X_train.columns) - set(X_test.columns)
    for c in missing:
        X_test[c] = 0
    return X_train, X_test[X_train.columns]


def compute_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")
    try:
        ll = log_loss(y_true, y_prob)
    except:
        ll = float("nan")
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC":   round(auc, 4),
        "Log Loss":  round(ll, 4),
    }


# ── Master test set (20% hold-out from original) ──
print("=" * 60)
print("  Comprehensive Classifier Evaluation")
print("=" * 60)
print("\nLoading held-out test set from original data...")
X_orig, y_orig = load_and_preprocess(DATASETS["Original"])
_, X_test_raw, _, y_test = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
)
print(f"Test set: {len(y_test)} samples (fraud=1: {(y_test==1).sum()}, not-fraud=0: {(y_test==0).sum()})\n")

# ── Train & evaluate all classifiers on all datasets ──
all_results = []

for ds_name, ds_path in DATASETS.items():
    print(f"\n--- Dataset: {ds_name} ---")
    X_train_raw, y_train = load_and_preprocess(ds_path)
    X_train_raw, X_test_aligned = align_features(X_train_raw, X_test_raw.copy())

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp  = imputer.transform(X_test_aligned)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test  = scaler.transform(X_test_imp).astype(np.float32)

    for clf_name, clf in CLASSIFIERS.items():
        print(f"  Training {clf_name}...", end=" ", flush=True)
        clf_instance = clf.__class__(**clf.get_params())
        clf_instance.fit(X_train, y_train)

        # Save model to disk
        safe_ds_name = ds_name.lower().replace(" ", "_").replace("-", "_")
        safe_clf_name = clf_name.lower().replace(" ", "_").replace("_(mlp)", "").replace("(", "").replace(")", "")
        model_filename = f"{safe_clf_name}_{safe_ds_name}.pkl"
        model_dir = os.path.join(ROOT, "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(clf_instance, os.path.join(model_dir, model_filename))

        y_pred = clf_instance.predict(X_test)
        try:
            y_prob = clf_instance.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = y_pred.astype(float)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        all_results.append({"Dataset": ds_name, "Classifier": clf_name, **metrics})
        print(f"Acc={metrics['Accuracy']}  F1={metrics['F1-Score']}  AUC={metrics['AUC-ROC']}")

# ── Results table ──
print(f"\n{'='*60}")
print("  FULL RESULTS TABLE")
print(f"{'='*60}")
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

csv_path = os.path.join(OUT_DIR, "classifier_comparison.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved -> {csv_path}")

# ── Charts ──
if plt is not None and sns is not None:
    sns.set_theme(style="whitegrid", palette="muted")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(24, 6))
    for ax, metric in zip(axes, metric_cols):
        sns.barplot(data=results_df, x="Dataset", y=metric, hue="Classifier", ax=ax)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=7, title=None)
    plt.suptitle("Classifier Performance Across Datasets", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "classifier_metrics_comparison.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out1}")
    plt.close()

    pivot_f1 = results_df.pivot(index="Classifier", columns="Dataset", values="F1-Score")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_f1, annot=True, fmt=".4f", cmap="YlGnBu",
                linewidths=0.5, cbar_kws={"label": "F1-Score"})
    plt.title("F1-Score Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "f1_heatmap.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out2}")
    plt.close()

    pivot_auc = results_df.pivot(index="Classifier", columns="Dataset", values="AUC-ROC")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot_auc, annot=True, fmt=".4f", cmap="RdYlGn",
                linewidths=0.5, cbar_kws={"label": "AUC-ROC"})
    plt.title("AUC-ROC Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "aucroc_heatmap.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out3}")
    plt.close()
else:
    print("Plotting skipped because matplotlib/seaborn are not installed.")

print("\nDone! All results saved to outputs/")
