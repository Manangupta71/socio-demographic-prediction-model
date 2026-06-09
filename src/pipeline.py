import os
import bandicoot as bc
import pandas as pd

def run_raw_feature_pipeline(raw_folder_path):
    """
    Ingests raw CDR files using bandicoot and compiles them 
    into the high-dimensional feature schema found in phone_features.csv.
    """
    print(f"[INFO] Initializing bandicoot feature extraction engine on: '{raw_folder_path}'")
    
    if not os.path.exists(raw_folder_path):
        raise FileNotFoundError(f"Directory '{raw_folder_path}' does not exist.")
        
    extracted_records = []
    phone_identifiers = []
    
    # 1. Loop through each user's raw phone log file
    for file_name in os.listdir(raw_folder_path):
        if file_name.endswith('.csv'):
            # The filename matches the unique phone identifier
            phone_no = file_name.replace('.csv', '')
            phone_identifiers.append(phone_no)
            
            # 2. Ingest transactional logs into a structured bandicoot container
            user_cdrs = bc.load(phone_no, raw_folder_path, recompute_virtual_records=True)
            
            # 3. Compute the full extended suite of behavior metrics
            # This handles the complex calculations for temporal patterns, day/night ratios,
            # contact entropy, and call duration distributions.
            metrics = bc.utils.all(user_cdrs, summary='extended')
            extracted_records.append(metrics)
            
    # 4. Flatten the dictionary metrics into a standardized machine learning matrix (X)
    feature_matrix = pd.DataFrame(extracted_records)
    
    # Insert the phone identifier column right at the front of the array
    feature_matrix.insert(0, 'phone_number', phone_identifiers)
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Feature extraction pipeline complete.")
    print(f" -> Unique profiles engineered: {feature_matrix.shape[0]}")
    print(f" -> Metrics compiled per user : {feature_matrix.shape[1] - 1}")
    print("="*60 + "\n")
    
    return feature_matrix

if __name__ == '__main__':
    # Production directory pointing to raw logs
    RAW_DATA_DIR = os.path.join('raw_data', 'sample_user_logs')
    
    # If the mock directory doesn't exist yet, we create it
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    print(f"[INFO] Gateway ready. Place your raw CDR log files inside '{RAW_DATA_DIR}'")