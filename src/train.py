"""
Multi-target socio-economic demographic prediction from CDR-derived mobility
+ graph features.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from pipeline import run_feature_alignment_pipeline

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

TARGET_ATTRIBUTES = [
    "socio_demographic_class", "age_group", "gender",
    "work_status", "household_auto", "driving_license", "income_bracket"
]

CATBOOST_PARAM_DIST = {
    "depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    "iterations": [150, 250, 400],
}

def train_and_evaluate_target(X, y, target_name, n_splits=5):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_accs, fold_f1s = [], []
    last_report = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        min_class_count = np.min(np.bincount(y_train))
        k_neighbors = max(1, min(5, min_class_count - 1))

        if min_class_count > 1:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        base_model = CatBoostClassifier(
            random_seed=RANDOM_STATE,
            verbose=0,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
        )

        search = RandomizedSearchCV(
            base_model,
            param_distributions=CATBOOST_PARAM_DIST,
            n_iter=6,
            cv=3,
            scoring="accuracy",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        search.fit(X_res, y_res)
        model = search.best_estimator_

        y_pred = model.predict(X_test).flatten()

        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average="macro"))

        if fold_idx == n_splits - 1:
            last_report = classification_report(
                y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0
            )

    return fold_accs, fold_f1s, last_report

def run_mumbai_multi_target_synthesis():
    print("\n" + "=" * 70)
    print("   MULTI-TARGET SOCIO-ECONOMIC DEMOGRAPHIC PREDICTION (MUMBAI)")
    print("=" * 70 + "\n")

    phone_ids, X_features = run_feature_alignment_pipeline(use_graph_features=True)

    survey_path = "raw_data/mumbai_survey.csv"
    survey_df = pd.read_csv(survey_path, dtype={"phone_number": str})
    X_features["phone_number"] = X_features["phone_number"].astype(str)

    merged_df = pd.merge(X_features, survey_df, on="phone_number", how="inner")
    print(f"[INGEST] Unified matrix: {merged_df.shape[0]} agents, {merged_df.shape[1]} columns.\n")

    exclude_cols = set(TARGET_ATTRIBUTES) | {"phone_number", "residential_ward"}
    feature_cols = [c for c in merged_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(merged_df[c])]
    
    X_matrix = merged_df[feature_cols].values

    results = {}
    for target in TARGET_ATTRIBUTES:
        print("-" * 70)
        print(f"Target: {target}")
        y_labels = merged_df[target].astype(str).values
       
        fold_accs, fold_f1s, last_report = train_and_evaluate_target(X_matrix, y_labels, target)

        mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
        mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
        results[target] = (mean_acc, std_acc, mean_f1, std_f1)

        print(f"  CV Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f})")
        print(f"  CV Macro-F1: {mean_f1:.3f}")

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for target, (mean_acc, std_acc, mean_f1, std_f1) in results.items():
        print(f"  {target:28s}  acc={mean_acc*100:5.2f}% (+/-{std_acc*100:4.2f})  macroF1={mean_f1:.3f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_mumbai_multi_target_synthesis()