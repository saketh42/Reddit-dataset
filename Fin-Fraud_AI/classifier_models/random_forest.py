"""
Random Forest Classifier
Trains on all 3 datasets and evaluates with full metrics.
"""
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from utils import DATASETS, get_test_set, prepare_data, compute_metrics

print("=" * 60)
print("  Random Forest — Training & Evaluation")
print("=" * 60)

model = RandomForestClassifier(n_estimators=100, random_state=42)
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
