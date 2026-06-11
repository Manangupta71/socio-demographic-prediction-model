import os
import glob
import numpy as np
import pandas as pd

def extract_bandicoot_spatial_features(raw_cdr_df):
    """
    Replicates the core mathematical feature extraction engine of the 
    standardized Bandicoot schema using optimized NumPy operations.
    """
    if raw_cdr_df.empty:
        return [0.0, 0.0, 0.0, 0.0, 1]
        
    # 1. Calculate Radius of Gyration (Rg)
    # Formulated as the root-mean-square error from the spatial trajectory's center of mass
    lat_cm = raw_cdr_df['latitude'].mean()
    lon_cm = raw_cdr_df['longitude'].mean()
    distances_squared = (raw_cdr_df['latitude'] - lat_cm)**2 + (raw_cdr_df['longitude'] - lon_cm)**2
    radius_of_gyration = np.sqrt(distances_squared.mean())
    
    # 2. Calculate Location Entropy (E)
    # Measures the predictability and spatial dispersion across active graph nodes
    node_probabilities = raw_cdr_df['tower_id'].value_counts(normalize=True).values
    location_entropy = -np.sum(node_probabilities * np.log2(node_probabilities + 1e-9))
    
    # 3. Calculate Temporal Peak Commute Ratio
    # Quantifies strict office hours vs localized midday patterns
    raw_cdr_df['timestamp'] = pd.to_datetime(raw_cdr_df['timestamp'])
    hours = raw_cdr_df['timestamp'].dt.hour
    
    peak_pings = ((hours >= 8) & (hours <= 10)) | ((hours >= 17) & (hours <= 19))
    off_peak_pings = (hours >= 12) & (hours <= 15)
    
    total_relevant = peak_pings.sum() + off_peak_pings.sum()
    peak_commute_ratio = peak_pings.sum() / total_relevant if total_relevant > 0 else 0.5
    
    # 4. Calculate Mass Transit Infrastructure Proximity
    # Computes network node coupling boundaries (True if tower sits near stations/metros)
    transit_proximity = raw_cdr_df['near_railway_node'].mean() if 'near_railway_node' in raw_cdr_df.columns else 0.4
    
    # 5. Extract Count Descriptors
    unique_anchors = raw_cdr_df['tower_id'].nunique()
    
    return [radius_of_gyration, location_entropy, peak_commute_ratio, transit_proximity, unique_anchors]


def run_feature_alignment_pipeline():
    """
    Main ingestion orchestrator. Checks for raw individual subscriber logs, 
    extracts graph metrics, or securely falls back to the master feature file.
    """
    print("[PIPELINE] Initializing feature extraction runtime framework...")
    
    raw_logs_pattern = os.path.join('raw_data', 'subscriber_logs', '*.csv')
    raw_log_files = glob.glob(raw_logs_pattern)
    
    # If raw log sheets exist in the workspace, run the feature extraction loops
    if len(raw_log_files) > 0:
        print(f"[EXTRACT] Found {len(raw_log_files)} raw user logs. Computing graph parameters...")
        extracted_records = []
        
        for file_path in raw_log_files:
            phone_num = os.path.basename(file_path).replace('.csv', '')
            df_log = pd.read_csv(file_path)
            
            # Extract metrics
            metrics = extract_bandicoot_spatial_features(df_log)
            extracted_records.append([phone_num] + metrics)
            
        feature_cols = ['phone_number', 'radius_of_gyration', 'location_entropy', 'peak_commute_ratio', 'rail_transit_proximity', 'unique_anchors']
        extracted_features_df = pd.DataFrame(extracted_records, columns=feature_cols)
        
        # Save output so it aligns with downstream ML steps
        extracted_features_df.to_csv('raw_data/mumbai_phone_features.csv', index=False)
        print("[EXTRACT] Feature processing complete. Output saved to 'raw_data/mumbai_phone_features.csv'.")
        return extracted_features_df['phone_number'].values, extracted_features_df
        
    # Default Fallback: Read the pre-aggregated structural schema if raw files aren't split
    else:
        print("[PIPELINE] Raw individual subscriber folders not found. Importing master feature schema matrix.")
        master_feature_path = os.path.join('raw_data', 'mumbai_phone_features.csv')
        
        if not os.path.exists(master_feature_path):
            raise FileNotFoundError(f"Missing master feature file at: {master_feature_path}")
            
        df_features = pd.read_csv(master_feature_path)
        
        # Audit data health & perform imputation
        df_features = df_features.fillna(df_features.median(numeric_only=True))
        phone_ids = df_features['phone_number'].values
        
        return phone_ids, df_features