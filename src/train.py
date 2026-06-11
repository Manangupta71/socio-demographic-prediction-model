import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from pipeline import run_feature_alignment_pipeline

def run_mumbai_multi_target_synthesis():
    print("\n" + "="*70)
    print("   LAUNCHING MULTI-OUTPUT AGENT DISAGGREGATE INFERENCE ENGINE")
    print("="*70 + "\n")
    
    # Ingest pre-computed mobility graph features
    phone_ids, X_features = run_feature_alignment_pipeline()
    
    # Ingest Full Vector Ground Truth Surveys
    survey_path = os.path.join('raw_data', 'mumbai_survey.csv')
    if not os.path.exists(survey_path):
        raise FileNotFoundError("Missing full survey matrices. Execute the updated data generator first.")
    survey_df = pd.read_csv(survey_path)
    
    # Perform relational join across the tracking key
    merged_df = pd.merge(X_features, survey_df, on='phone_number', how='inner')
    print(f"[INGEST] Unified relational matrix matching complete. Seed Pool: {merged_df.shape[0]} Agents.")
    
    # Explicitly pull target array listings from your schema document
    target_attributes = [
        'socio_demographic_class', 'age_group', 'gender', 
        'work_status', 'household_auto', 'driving_license', 'income_bracket'
    ]
    
    # Isolate independent analytical graph variables
    feature_cols = ['radius_of_gyration', 'location_entropy', 'peak_commute_ratio', 'rail_transit_proximity', 'unique_anchors']
    X_matrix = merged_df[feature_cols].values
    
    print("\n" + "="*70)
    print("          MUMBAI SYNTHETIC POPULATION PIPELINE EVALUATION")
    print("="*70)
    
    # Loop through each target attribute to train discrete specialized estimators
    for target in target_attributes:
        print(f"\n⏳ Optimizing classification layer for target element: [{target}]")
        y_labels = merged_df[target].astype(str).values
        
        # Partition data blocks
        X_train, X_test, y_train, y_test = train_test_split(
            X_matrix, y_labels, 
            test_size=0.20, 
            random_state=42, 
            stratify=y_labels
        )
        
        # Balance sample space via SMOTE to neutralize potential expansion bias
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Setup and train structural CatBoost estimator
        model = CatBoostClassifier(
            iterations=250,
            learning_rate=0.06,
            depth=5,
            loss_function='MultiClass' if len(np.unique(y_labels)) > 2 else 'Logloss',
            random_seed=42,
            verbose=0 # Suppress internal iterations print to keep output perfectly clean
        )
        
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predict validation pool
        y_pred = model.predict(X_test).flatten()
        acc = accuracy_score(y_test, y_pred)
        
        print(f"✅ Target [{target}] - Synthesis Accuracy: {acc * 100:.2f}%")
        print(f"--- Micro-Validation Report for {target} ---")
        print(classification_report(y_test, y_pred))
        print("-" * 70)
        
    print("\n" + "="*70)
    print("  MULTI-OUTPUT POPULATION INFRASTRUCTURE VALIDATED FOR IPF EXPANSION")
    print("="*70 + "\n")

if __name__ == '__main__':
    run_mumbai_multi_target_synthesis()