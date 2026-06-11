# Mumbai Socio-Demographic Profile Inference Framework

## Overview
This framework uses synthetic mobile phone Call Detail Records (CDRs) to predict multi-class socio-demographic and wealth-related strata across the dense urban topography of Mumbai, India. By passing spatial path descriptors (such as *Radius of Gyration* and *Location Entropy*) through a depth-constrained gradient boosting tree ensemble, the model infers a subscriber's socioeconomic standing based on their physical travel routines.

## Project Structure
* `src/generate_mumbai_data.py` : Synthetic data generator engine utilizing spatio-temporal Markov chains and human Explore-Return (XR) mechanics to establish realistic Mumbai local train/vehicular daily travel chains.
* `src/pipeline.py` : Decoupled data ingestion layer responsible for loading feature spaces, checking vector data health, and feeding balanced matrices to the model.
* `src/train.py` : Core machine learning execution engine executing data fusion database joins, SMOTE class balancing, and a CatBoost tree classifier optimized under cross-validation loops.
* `raw_data/` : Local folder housing the generated Mumbai features and survey answer sheets.

## Setup & Environment Verification

This framework is built and optimized for **Python 3.10**.

1. Create and activate your isolated virtual environment:
   ```bash
   python -m venv .venv

```

*Windows (PowerShell):*

```bash
.venv\Scripts\Activate.ps1

```

2. Force-install the required machine learning and data engineering binaries directly inside your local environment path:
```bash
& .venv/Scripts/python.exe -m pip install pandas numpy catboost scikit-learn imbalanced-learn

```



## Execution Workflow Sequence

To run the complete end-to-end framework, execute the following scripts in order from your terminal root:

1. **Step 1: Generate the Localized Mumbai Trajectory Dataset**
```bash
python src/generate_mumbai_data.py

```


*This compiles 1,000 unique subscriber profiles mapped to Mumbai's geographic transit hubs inside the `raw_data/` folder.*
2. **Step 2: Trigger the Feature Ingestion & Audit Pipeline**
```bash
python src/pipeline.py

```


*Verifies path bounds, screens for corrupted null vectors, and tests alignment health.*
3. **Step 3: Execute Model Training and Policy Risk Metrics Evaluation**
```bash
python src/train.py

```


*Executes SMOTE balancing, fits the CatBoost Classifier, and prints out your Multiclass Classification Accuracy, Policy Exclusion Errors, and Policy Inclusion Errors.*

## Engineering Pipeline Scalability

The architecture of this project is completely decoupled. Because all feature extraction processing layers are structurally isolated inside `src/pipeline.py`, this entire simulation platform is fully production-ready. Once authentic corporate telecom CDR metadata records and boots-on-the-ground regional household census surveys are acquired, they can be plugged directly into the `raw_data/` folder path to update the model splits without rewriting any core classification backend logic.
