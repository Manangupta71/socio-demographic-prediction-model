# Socio-Demographic Prediction Model 
"# socio-demographic-prediction-model" 
## Data Acknowledgment
This project leverages the pre-engineered, high-dimensional anonymized mobile feature matrices made available by the replication framework for:

Aiken, E. L., Bedoya, G., Blumenstock, J. E., & Coville, A. (2023). 
"Program targeting with machine learning and mobile phone data: Evidence from an anti-poverty intervention in Afghanistan." 
*Journal of Development Economics*, 161.

The input feature engine matrix is structured to mimic the standardized schemas utilized in the open-source `bandicoot` toolbox.

## Overview

This project predicts socio-demographic and wealth-related indicators using mobile phone Call Detail Records (CDRs).

## Project Structure

* `src/` : Source code
* `data/` : Input datasets
* `notebooks/` : Experiments and analysis notebooks

## Setup

This project requires **Python 3.10**.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

```bash
python src/pipeline.py
```

## Important Note

The project uses the **Bandicoot** library for CDR feature extraction. Bandicoot is not compatible with Python 3.13 and should be run using Python 3.10.

