"""
Neural Network (MLP) Classifier
Trains on all 3 datasets and evaluates with full metrics.
"""
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from utils import DATASETS, get_test_set, prepare_data, compute_metrics

print("=" * 60)
print("  Neural Network (MLP) — Training & Evaluation")
print("=" * 60)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=300,
    random_state=42,
    early_stopping=True
)
X_test_raw, y_test = get_test_set()

for ds_name, ds_path in DATASETS.items():
    X_train, y_train, X_test = prepare_data(ds_path, X_test_raw)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print(f"\n--- {ds_name} ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

print("\nDone!")
