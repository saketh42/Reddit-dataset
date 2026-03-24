import os
import sys
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", "outputs", ".mplconfig"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classifier_models.evaluation_core import DATASETS, OUT_DIR, get_master_test_split, prepare_scaled_data

EPSILON = 0.2
PGD_STEPS = 40
PGD_STEP_SIZE = 0.02


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_input_gradient(model, X, y_true):
    probs = sigmoid(X @ model.coef_.ravel() + model.intercept_[0])
    return ((probs - y_true)[:, None] * model.coef_).astype(np.float32)


def fgsm_attack(model, X, y_true, epsilon=EPSILON):
    grad = logistic_input_gradient(model, X, y_true)
    return (X + epsilon * np.sign(grad)).astype(np.float32)


def pgd_attack(model, X, y_true, epsilon=EPSILON, step_size=PGD_STEP_SIZE, steps=PGD_STEPS):
    X_start = X.copy()
    X_adv = X.copy()
    for _ in range(steps):
        grad = logistic_input_gradient(model, X_adv, y_true)
        X_adv = X_adv + step_size * np.sign(grad)
        delta = np.clip(X_adv - X_start, -epsilon, epsilon)
        X_adv = (X_start + delta).astype(np.float32)
    return X_adv


def evaluate_dataset(name, train_path, test_X_raw, test_y):
    print(f"\n--- Evaluating {name} ---")
    X_train, y_train, X_test, _ = prepare_scaled_data(train_path, test_X_raw)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    clean_pred = model.predict(X_test)
    clean_acc = accuracy_score(test_y, clean_pred)

    X_fgsm = fgsm_attack(model, X_test, test_y)
    fgsm_pred = model.predict(X_fgsm)
    fgsm_acc = accuracy_score(test_y, fgsm_pred)
    fgsm_perturb = float(np.mean(np.linalg.norm(X_fgsm - X_test, axis=1)))

    X_pgd = pgd_attack(model, X_test, test_y)
    pgd_pred = model.predict(X_pgd)
    pgd_acc = accuracy_score(test_y, pgd_pred)
    pgd_perturb = float(np.mean(np.linalg.norm(X_pgd - X_test, axis=1)))

    robustness_score = clean_acc - min(fgsm_acc, pgd_acc)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"FGSM Accuracy:  {fgsm_acc:.4f} | Perturbation: {fgsm_perturb:.4f}")
    print(f"PGD Accuracy:   {pgd_acc:.4f} | Perturbation: {pgd_perturb:.4f}")
    print(f"Robustness Score: {robustness_score:.4f}")

    return {
        "Dataset": name,
        "Clean Accuracy": round(clean_acc, 4),
        "FGSM Accuracy": round(fgsm_acc, 4),
        "PGD Accuracy": round(pgd_acc, 4),
        "Robustness Score": round(robustness_score, 4),
        "FGSM Perturbation": round(fgsm_perturb, 4),
        "PGD Perturbation": round(pgd_perturb, 4),
    }


def main():
    print("Loading Master Test Set (Held-out from Original Data)")
    _, X_test_raw, _, y_test = get_master_test_split()

    results = []
    name_map = {
        "Original": "Original (Imbalanced)",
        "CTGAN": "Standard CTGAN",
        "Adv-CTGAN": "Adv-CTGAN (Custom)",
    }

    for ds_key, ds_path in DATASETS.items():
        results.append(evaluate_dataset(name_map[ds_key], ds_path, X_test_raw.copy(), y_test))

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "robustness_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        melt_acc = results_df.melt(
            id_vars="Dataset",
            value_vars=["Clean Accuracy", "FGSM Accuracy", "PGD Accuracy"],
            var_name="Condition",
            value_name="Accuracy",
        )
        sns.barplot(data=melt_acc, x="Dataset", y="Accuracy", hue="Condition", ax=axes[0])
        axes[0].set_title("Accuracy: Clean vs Adversarial Attacks", fontweight="bold")
        axes[0].set_ylim(0, 1.1)

        melt_rob = results_df.melt(
            id_vars="Dataset",
            value_vars=["Robustness Score", "FGSM Perturbation", "PGD Perturbation"],
            var_name="Metric",
            value_name="Value",
        )
        sns.barplot(data=melt_rob, x="Dataset", y="Value", hue="Metric", ax=axes[1])
        axes[1].set_title("Robustness Score and Perturbation", fontweight="bold")

        plt.suptitle("Adversarial Robustness Evaluation", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "robustness_comparison.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out_png}")
        plt.close()
    else:
        print("Plotting skipped because matplotlib/seaborn are not installed.")


if __name__ == "__main__":
    main()
