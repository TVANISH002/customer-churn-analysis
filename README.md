## Customer-Churn-Analysis

## Overview
This project implements an **end-to-end machine learning solution** to predict whether a bank customer is likely to **leave the bank (churn)**.  
It focuses on **clean data preprocessing, model training, and inference-ready deployment**, using an **Artificial Neural Network (ANN)**.

The repository demonstrates practical skills in:
- Tabular data preprocessing
- Binary classification modeling
- Model persistence and reproducible inference


## Business Problem
Customer churn directly impacts revenue in the banking sector.  
The goal of this project is to **predict churn risk** based on customer attributes, enabling early intervention and retention strategies.

**Target Variable:** `Exited`  
- `1` → Customer left the bank  
- `0` → Customer retained  


## Dataset
**Source:** `Churn_Modelling.csv`

Each record represents a bank customer with attributes such as:
- Credit score, age, tenure, balance
- Geography, gender
- Product usage and activity indicators

Non-predictive identifiers (`RowNumber`, `CustomerId`, `Surname`) are excluded during preprocessing.


## Technical Approach

### Data Preparation
- Cleaned and filtered raw customer data
- Encoded categorical variables:
  - Gender → Label Encoding
  - Geography → One-Hot Encoding
- Scaled numerical features using **StandardScaler**
- Split data into training and testing sets


### Model
An **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** for binary classification:

- Dense layer (64 units, ReLU)
- Dense layer (32 units, ReLU)
- Output layer (Sigmoid)

**Loss:** Binary Crossentropy  
**Optimizer:** Adam  
**Regularization:** EarlyStopping to prevent overfitting


### Model Outputs
- Generates a **probability score (0–1)** indicating churn likelihood
- Converts probability into a final churn decision


## Saved Artifacts
To support consistent inference and deployment, the following artifacts are stored:
- Trained ANN model (`model.h5`)
- Gender encoder (`label_encoder_gender.pkl`)
- Geography encoder (`onehot_encoder_geo.pkl`)
- Feature scaler (`scaler.pkl`)


## Repository Structure
- `experiments.ipynb` – Data preprocessing, feature engineering, and model training  
- `prediction.ipynb` – Inference using saved artifacts  
- `app.py` – Inference application logic  
- `Churn_Modelling.csv` – Dataset  
- `model.h5` – Trained ANN model  
- `scaler.pkl` – StandardScaler  
- `label_encoder_gender.pkl` – LabelEncoder  
- `onehot_encoder_geo.pkl` – OneHotEncoder  
- `requirements.txt` – Dependencies  

