import os
import pandas as pd
import numpy as np

def run_feature_alignment_pipeline(features_path=None):
    """
    Production-ready data pipeline. It ingests the high-dimensional behavioral 
    matrix, isolates identifiers, and sanitizes features by computing column-wise 
    means to handle missing statistical values out of the box.
    """
    print("[PIPELINE] Initializing data ingestion wrapper...")
    
    if features_path is None:
        features_path = os.path.join('raw_data', 'phone_features.csv')
        
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Data entry checkpoint failed. Could not locate: '{features_path}'. "
            "Please ensure phone_features.csv is placed inside your 'raw_data/' folder."
        )
        
    # Read the data matrix containing 800+ bandicoot behavioral attributes
    feature_matrix = pd.read_csv(features_path)
    
    # Extract unique subscriber phone keys
    phone_identifiers = feature_matrix['phone_number']
    
    # Isolate structural predictors (X)
    X_raw = feature_matrix.drop(columns=['phone_number'], errors='ignore')
    
    # Handle numeric columns containing NaN values (e.g., skewness/kurtosis on low interaction counts)
    # Replaces missing entries dynamically with the statistical mean of that specific behavioral metric
    print("[PIPELINE] Engineering data sanity checks and mean imputation...")
    X_clean = X_raw.fillna(X_raw.mean())
    
    # Final check: If a column is completely empty, fall back to filling with 0
    X_clean = X_clean.fillna(0)
    
    print("=" * 60)
    print("[SUCCESS] Pipeline Feature Matrix Extraction Complete")
    print(f"  * Total Active Profiles Processed : {X_clean.shape[0]}")
    print(f"  * Engineered Attributes Extracted : {X_clean.shape[1]}")
    print("=" * 60 + "\n")
    
    return phone_identifiers, X_clean

if __name__ == '__main__':
    # Test execution rule to verify pipeline compilation independently
    try:
        ids, features = run_feature_alignment_pipeline()
        print("Verification Matrix Sample (First 5 users, first 3 features):")
        print(features.iloc[:5, :3])
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")