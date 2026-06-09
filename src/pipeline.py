import os
import pandas as pd
import numpy as np

# Try to import bandicoot smoothly. If it's not installed on the system yet, 
# the pipeline will still run safely using fallback development modes.
try:
    import bandicoot as bc
    BANDICOOT_AVAILABLE = True
except ImportError:
    BANDICOOT_AVAILABLE = False

def run_feature_alignment_pipeline(raw_logs_dir=None, fallback_csv_path=None):
    """
    A true plug-and-play production pipeline gateway. 
    1. If a directory of raw text/call logs is provided, it parses them with bandicoot.
    2. If no raw folder is found, it automatically switches to the pre-computed matrix.
    """
    # Assign default workspace paths if nothing is passed explicitly
    if raw_logs_dir is None:
        raw_logs_dir = os.path.join('raw_data', 'sample_user_logs')
    if fallback_csv_path is None:
        fallback_csv_path = os.path.join('raw_data', 'phone_features.csv')

    # -------------------------------------------------------------------------
    # SCENARIO A: Dynamic Plug-and-Play Raw Processing Mode
    # -------------------------------------------------------------------------
    # Check if a folder exists and contains raw user transaction files
    if os.path.exists(raw_logs_dir) and any(f.endswith('.csv') for f in os.listdir(raw_logs_dir)):
        if not BANDICOOT_AVAILABLE:
            print("[WARN] Raw logs detected, but 'bandicoot' library is missing!")
            print("       Falling back directly to pre-computed matrix strategy...")
        else:
            print(f"[PIPELINE-RAW] Live data stream detected in '{raw_logs_dir}'")
            print("[PIPELINE-RAW] Initializing dynamic bandicoot extraction engine...")
            
            extracted_profiles = []
            phone_identifiers = []
            
            for file_name in os.listdir(raw_logs_dir):
                if file_name.endswith('.csv'):
                    phone_no = file_name.replace('.csv', '')
                    phone_identifiers.append(phone_no)
                    
                    # Core bandicoot calls: load raw primitive rows & extract extended metrics
                    user_records = bc.load(phone_no, raw_logs_dir, recompute_virtual_records=True)
                    metrics = bc.utils.all(user_records, summary='extended')
                    extracted_profiles.append(metrics)
            
            # Flatten the multi-layered dictionary arrays into our feature matrix (X)
            X_raw = pd.DataFrame(extracted_profiles)
            X_clean = X_raw.fillna(X_raw.mean()).fillna(0)
            
            _print_success_banner(X_clean.shape[0], X_clean.shape[1], data_type="LIVE CDR LOGS")
            return pd.Series(phone_identifiers, name='phone_number'), X_clean

    # -------------------------------------------------------------------------
    # SCENARIO B: Research/Development Fallback Mode (The Pre-computed Array)
    # -------------------------------------------------------------------------
    print(f"[PIPELINE-STATIC] No raw streams active. Loading fallback source: '{fallback_csv_path}'")
    if not os.path.exists(fallback_csv_path):
        raise FileNotFoundError(
            f"Pipeline Gateway Failure. Could not resolve raw folders or fallback file: '{fallback_csv_path}'"
        )
        
    feature_matrix = pd.read_csv(fallback_csv_path)
    phone_identifiers = feature_matrix['phone_number']
    X_raw = feature_matrix.drop(columns=['phone_number'], errors='ignore')
    
    # Run missing value safety checks
    X_clean = X_raw.fillna(X_raw.mean()).fillna(0)
    
    _print_success_banner(X_clean.shape[0], X_clean.shape[1], data_type="PRE-COMPUTED MATRIX")
    return phone_identifiers, X_clean


def _print_success_banner(rows, cols, data_type):
    """Prints a clean status window inside the command prompt."""
    print("\n" + "=" * 65)
    print(f"[SUCCESS] PIPELINE PROCESSING COMPLETED")
    print(f"  * Ingestion Influx Model : {data_type}")
    print(f"  * Total Profiles Matched : {rows}")
    print(f"  * Total Active Features  : {cols}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    # Test script execution rules to make sure it handles files out of the box
    try:
        ids, features = run_feature_alignment_pipeline()
        print("Sample Layout Head:\n", features.iloc[:3, :3])
    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline breakdown: {e}")