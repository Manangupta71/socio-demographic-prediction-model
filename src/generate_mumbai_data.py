import os
import numpy as np
import pandas as pd

def generate_full_schema_mumbai_dataset():
    print("\n" + "="*70)
    print("   GENERATING MULTI-TARGET DISAGGREGATE POPULATION ARRAYS")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 1000
    
    classes = ['Type_a', 'Type_b', 'Type_c', 'Type_d', 'Type_e']
    archetypes = np.random.choice(classes, size=n_samples, p=[0.15, 0.20, 0.35, 0.20, 0.10])
    
    phone_numbers = [f"+919820{i:06d}" for i in range(n_samples)]
    
    features = []
    survey_records = []
    
    for arch in archetypes:
        # Default baseline initializations
        gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48]) # Base Mumbai distribution
        
        if arch == 'Type_a':    # Adolescent / Student
            rg = np.random.uniform(5.0, 12.0)
            entropy = np.random.uniform(1.5, 2.5)
            peak_ratio = np.random.uniform(0.60, 0.85)
            transit_proximity = np.random.uniform(0.40, 0.70)
            
            age_group = '0-22'
            work_status = 'Unemployed_or_Retired' # Subsumed by school
            household_auto = np.random.choice(['0 cars', '1_or_more'], p=[0.70, 0.30])
            driving_license = np.random.choice(['Yes', 'No'], p=[0.20, 0.80])
            income_bracket = 'Low_Income'
            
        elif arch == 'Type_b':  # Middle-Aged, Employed, Car Convenience
            rg = np.random.uniform(18.0, 32.0)
            entropy = np.random.uniform(2.0, 3.2)
            peak_ratio = np.random.uniform(0.75, 0.92)  
            transit_proximity = np.random.uniform(0.20, 0.55)
            
            age_group = '22-60'
            work_status = 'Steady_Jobs_or_School'
            household_auto = '1_or_more'
            driving_license = 'Yes'
            income_bracket = 'Middle_High_Income'
            
        elif arch == 'Type_c':  # Middle-Aged, Employed, Public Transit Dependent
            rg = np.random.uniform(15.0, 35.0)  
            entropy = np.random.uniform(1.8, 3.0)
            peak_ratio = np.random.uniform(0.80, 0.95)  
            transit_proximity = np.random.uniform(0.45, 0.85)
            
            age_group = '22-60'
            work_status = 'Steady_Jobs_or_School'
            household_auto = np.random.choice(['0 cars', '1_or_more'], p=[0.85, 0.15])
            driving_license = np.random.choice(['Yes', 'No'], p=[0.40, 0.60])
            income_bracket = np.random.choice(['Low_Income', 'Middle_High_Income'], p=[0.60, 0.40])
            
        elif arch == 'Type_d':  # Informal Sector Worker / Unemployed
            rg = np.random.uniform(2.0, 6.0)
            entropy = np.random.uniform(3.2, 4.8)
            peak_ratio = np.random.uniform(0.30, 0.60)  
            transit_proximity = np.random.uniform(0.30, 0.65)
            
            age_group = '22-60'
            work_status = 'Unemployed_or_Retired'
            household_auto = '0 cars'
            driving_license = np.random.choice(['Yes', 'No'], p=[0.10, 0.90])
            income_bracket = 'Low_Income'
            
        else:                   # Elderly / Retired Population
            rg = np.random.uniform(1.0, 5.0)
            entropy = np.random.uniform(1.0, 2.2)
            peak_ratio = np.random.uniform(0.15, 0.40)  
            transit_proximity = np.random.uniform(0.25, 0.55)
            
            age_group = '60_and_above'
            work_status = 'Unemployed_or_Retired'
            household_auto = np.random.choice(['0 cars', '1_or_more'], p=[0.60, 0.40])
            driving_license = np.random.choice(['Yes', 'No'], p=[0.50, 0.50])
            income_bracket = np.random.choice(['Low_Income', 'Middle_High_Income'], p=[0.50, 0.50])
            
        # Append features with stochastic noise variance
        features.append([
            max(0.1, rg + np.random.normal(0, 2.0)),
            max(0.1, entropy + np.random.normal(0, 0.3)),
            max(0.01, min(0.99, peak_ratio + np.random.normal(0, 0.07))),
            max(0.01, min(0.99, transit_proximity + np.random.normal(0, 0.10))),
            np.random.uniform(10, 50)
        ])
        
        # Append full survey demographics vector
        survey_records.append([
            arch, age_group, gender, work_status, household_auto, driving_license, income_bracket
        ])
        
    feature_cols = ['radius_of_gyration', 'location_entropy', 'peak_commute_ratio', 'rail_transit_proximity', 'unique_anchors']
    df_features = pd.DataFrame(features, columns=feature_cols)
    df_features.insert(0, 'phone_number', phone_numbers)
    
    survey_cols = ['socio_demographic_class', 'age_group', 'gender', 'work_status', 'household_auto', 'driving_license', 'income_bracket']
    df_survey = pd.DataFrame(survey_records, columns=survey_cols)
    df_survey.insert(0, 'phone_number', phone_numbers)
    
    os.makedirs('raw_data', exist_ok=True)
    df_features.to_csv('raw_data/mumbai_phone_features.csv', index=False)
    df_survey.to_csv('raw_data/mumbai_survey.csv', index=False)
    print(f"[SUCCESS] Complete multi-target demographic matrices compiled in 'raw_data/'.")

if __name__ == '__main__':
    generate_full_schema_mumbai_dataset()