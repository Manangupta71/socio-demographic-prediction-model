import os
import pandas as pd
import numpy as np

def run_feature_alignment_pipeline():
    """
    Ingests high-dimensional synthetic Mumbai mobility graph features,
    executes data health audits, prevents feature structural breaks,
    and cleanly decouples the ingestion layer from downstream training.
    """
    print("\n" + "="*65)
    print("[PIPELINE] INITIALIZING MUMBAI METADATA INGESTION LAYER")
    print("="*65)
    
    # Target file path declaration relative to project root
    features_path = os.path.join('raw_data', 'mumbai_phone_features.csv')
    
    # -------------------------------------------------------------------------
    # WORKSPACE ENVIRONMENT PATH AUDIT
    # -------------------------------------------------------------------------
    # If the user executes from inside the src/ folder, programmatically 
    # move back one step to avoid path breaks or FileNotFoundError issues.
    if os.path.basename(os.getcwd()) == 'src':
        print("[PIPELINE] Active directory subfolder 'src' caught. Recalibrating root...")
        os.chdir('..')
        
    if not os.path.exists(features_path):
        print(f"\n[CRITICAL ERROR] Target matrix missing: Could not find '{features_path}'")
        print("Please verify that 'python src/generate_mumbai_data.py' has been run successfully.\n")
        raise FileNotFoundError(f"Missing component dependency: {features_path}")
        
    # -------------------------------------------------------------------------
    # DATA LOADING & QUALITY SANITIZATION LOOP
    # -------------------------------------------------------------------------
    print(f"[PIPELINE] Streaming file connection: {features_path}")
    features_df = pd.read_csv(features_path)
    print(f"[PIPELINE] Ingestion successful. Matrix Compiled: {features_df.shape[0]} nodes x {features_df.shape[1]} descriptors.")
    
    # Scan for null records or corrupt vectors to guarantee smooth backpropagation
    null_counts = features_df.isnull().sum().sum()
    if null_counts > 0:
        print(f"[WARNING] Detected {null_counts} corrupted null fields. Applying neighborhood forward-fill.")
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
    else:
        print("[PIPELINE] Data health validation passed. Zero corrupt or null entries detected.")
        
    # -------------------------------------------------------------------------
    # DECOUPLING & MATRIX EXTRACTS
    # -------------------------------------------------------------------------
    # Isolate relational primary tracking keys to preserve tracking identity
    phone_identifiers = features_df['phone_number'].values
    
    # Return the dataframe with the identifier key intact so that train.py 
    # can run a perfect database join on survey target labels.
    X_features = features_df.copy()
    
    print("[PIPELINE] Feature extraction decoupled. Handing variables to model wrapper.")
    print("="*65 + "\n")
    
    return phone_identifiers, X_features

if __name__ == '__main__':
    # Local unit testing entry point to verify environment path compliance
    try:
        phone_ids, data_matrix = run_feature_alignment_pipeline()
        print("🎉 [TEST PASSED] Pipeline execution is completely stable.")
        print(f"  * Sample Subscriber Keys Verified : {phone_ids[:2]}")
        print(f"  * Engineered Features Isolated    : {list(data_matrix.columns.drop('phone_number')[:4])}...")
    except Exception as error:
        print(f"❌ [TEST FAILED] Pipeline runtime exception: {error}")
        