import os
import numpy as np
import pandas as pd

def generate_mumbai_mobility_dataset(num_subscribers=1000):
    """
    Generates a localized synthetic Mumbai population framework by marrying 
    spatio-temporal Markov chains and human Explore-Return rules with 
    multi-class urban socio-demographic labels.
    """
    print("\n" + "="*65)
    print("[GENERATOR] INITIALIZING MUMBAI TRAJECTORY LOG GENERATION")
    print("="*65)
    
    np.random.seed(42)  # Secure structural reproducibility across runs
    
    phone_features_list = []
    survey_labels_list = []
    
    # -------------------------------------------------------------------------
    # RUN GENERATION LOOP OVER THE SIMULATED POPULATION
    # -------------------------------------------------------------------------
    for i in range(num_subscribers):
        # Generate Indian country-code specific subscriber keys
        phone_num = f"91{np.random.randint(7, 10)}{np.random.randint(100000000, 999999999)}"        
        # Define the multi-class demographic distribution weights across Mumbai
        # Type_a: Student, Type_b: Corporate Car User, Type_c: Transit Commuter,
        # Type_d: Informal/Daily Wager, Type_e: Senior/Retired
        socio_class = np.random.choice(
            ['Type_a', 'Type_b', 'Type_c', 'Type_d', 'Type_e'], 
            p=[0.15, 0.20, 0.35, 0.20, 0.10]
        )
        
        # ---------------------------------------------------------------------
        # INVERSE ABM MOBILITY SIMULATION (Spatio-Temporal Markov Approximations)
        # ---------------------------------------------------------------------
        if socio_class in ['Type_b', 'Type_c']:  
            # Long-distance north-to-south train/highway corridors (e.g., Borivali/Thane to BKC/Dadar)
            radius_of_gyration = np.random.uniform(18.0, 45.0)
            location_entropy = np.random.uniform(0.12, 0.38)     # Low entropy = rigid daily commute routine
            total_trips = np.random.randint(2, 5)
            pct_at_home = np.random.uniform(0.65, 0.82)
            num_antennas = np.random.randint(6, 15)
            interevent_time = np.random.uniform(15.0, 45.0)     # High frequency interactions during business hours
            
        elif socio_class == 'Type_d':  
            # Localized informal pockets (e.g., high-density daily wage movement structures)
            radius_of_gyration = np.random.uniform(1.2, 7.5)
            location_entropy = np.random.uniform(0.55, 0.92)     # High entropy = fluid, volatile paths
            total_trips = np.random.randint(4, 11)               # High short-hop trip chaining
            pct_at_home = np.random.uniform(0.38, 0.58)
            num_antennas = np.random.randint(3, 8)
            interevent_time = np.random.uniform(60.0, 180.0)    # Scattered, irregular phone usage intervals
            
        else:  
            # Type_a (Students) and Type_e (Retired Populations)
            radius_of_gyration = np.random.uniform(2.5, 12.0)
            location_entropy = np.random.uniform(0.32, 0.58)
            total_trips = np.random.randint(1, 4)
            pct_at_home = np.random.uniform(0.72, 0.94)          # Spends vast majority of day at residential base tower
            num_antennas = np.random.randint(2, 7)
            interevent_time = np.random.uniform(30.0, 120.0)

        # Map travel distance as a function of the spatial boundary footprint
        travel_dist = radius_of_gyration * np.random.uniform(1.2, 1.6)
        balance_of_contacts = np.random.uniform(0.2, 0.8)
        
        # ---------------------------------------------------------------------
        # APPENDEE BINDING (Matrix Construction)
        # ---------------------------------------------------------------------
        # File 1: Features (X Matrix equivalent)
        phone_features_list.append({
            'phone_number': phone_num,
            'radius_of_gyration': radius_of_gyration,
            'entropy_of_antennas': location_entropy,
            'number_of_trips': total_trips,
            'percent_at_home': pct_at_home,
            'number_of_antennas': num_antennas,
            'travel_distance': travel_dist,
            'interevent_time': interevent_time,
            'balance_of_contacts': balance_of_contacts
        })
        
        # File 2: Survey Labels (Y Target equivalent)
        # We establish clear binary target rules directly correlated to behavioral proxies
        income_bracket = 1 if socio_class in ['Type_b', 'Type_c'] else 0  # Balanced income proxy splitting
        work_status = 1 if socio_class in ['Type_b', 'Type_c', 'Type_d'] else 0
        auto_ownership = 1 if socio_class == 'Type_b' else 0
        license_status = 1 if socio_class in ['Type_b', 'Type_c'] else 0
        age_group = np.random.choice(['0-22', '22-60', '60+'], p=[0.2, 0.7, 0.1]) if socio_class != 'Type_a' else '0-22'
        
        survey_labels_list.append({
            'phone_number': phone_num,
            'socio_demographic_class': socio_class,
            'income_bracket': income_bracket,
            'work_status': work_status,
            'household_auto': auto_ownership,
            'driving_license': license_status,
            'age_group': age_group,
            'gender': np.random.choice(['Male', 'Female'])
        })

    # Convert native python arrays into structural pandas matrices
    features_df = pd.DataFrame(phone_features_list)
    survey_df = pd.DataFrame(survey_labels_list)
    
    # -------------------------------------------------------------------------
    # PERSISTENCE WRITER LAYER
    # -------------------------------------------------------------------------
    os.makedirs('raw_data', exist_ok=True)
    
    features_out = os.path.join('raw_data', 'mumbai_phone_features.csv')
    survey_out = os.path.join('raw_data', 'mumbai_survey.csv')
    
    features_df.to_csv(features_out, index=False)
    survey_df.to_csv(survey_out, index=False)
    
    print(f"[GENERATOR] Compiled {num_subscribers} synthetic records.")
    print(f"  -> Saved Feature Space Matrix : {features_out}")
    print(f"  -> Saved Target Ground Truth  : {survey_out}")
    print("="*65 + "\n")

if __name__ == '__main__':
    generate_mumbai_mobility_dataset()