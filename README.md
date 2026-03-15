
# Time-Series Energy Demand Forecasting System with Feature Drift Monitoring

A production-style machine learning service that performs **time-series forecasting** and **feature drift monitoring** for electricity demand data.

The system exposes a **FastAPI web application** that accepts batch feature datasets, returns predictions, and monitors whether incoming production data has **drifted from the training distribution**.

This project demonstrates key concepts required in real-world ML systems:

- Time-series modeling
- Production inference APIs
- Feature drift monitoring
- Rolling data monitoring
- Containerized deployment


## Project Overview

In real-world ML systems, model performance can degrade when **incoming production data differs from the training data**.

This project addresses that problem by:

1. Training a **time-series forecasting model** using lag-based features.
2. Deploying the model as a **FastAPI inference service**.
3. Accepting **CSV batch inference requests**.
4. Accumulating production inputs.
5. Detecting **feature distribution drift** over a rolling monitoring window.

If drift is detected, the system alerts the user that **model retraining may be required**.


## System Architecture

User CSV Upload  
&emsp;&emsp;&emsp;&emsp;↓  
FastAPI Inference Service  
&emsp;&emsp;&emsp;&emsp;↓  
Generate Predictions + Store Inputs  
&emsp;&emsp;&emsp;&emsp;↓  
Feature Drift Detection (rolling 500-row window)  
&emsp;&emsp;&emsp;&emsp;↓  
Drift Alert + Predictions Returned


## Model

The forecasting model predicts **electricity demand** using time-series features derived from historical data.

### Data
https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

### Model
XGBoost Regressor

### Feature Engineering

Features include lag and rolling statistics:

- lag_1
- lag_2
- lag_24
- lag_48
- rolling_mean_24
- rolling_std_24
- hour
- day_of_week
- month

These features allow the model to capture:

- short-term temporal dependencies
- daily seasonality
- volatility changes


## Feature Drift Detection

The system monitors whether incoming data diverges from the training distribution.

### Reference Distribution

Computed from:

data/processed/train_features.csv

For each feature we store:

- training mean
- training standard deviation

### Monitoring Window

Drift detection runs once the system has accumulated at least:

500 inference rows

The **latest 500 rows** are compared against the training distribution.

### Two Drift Checks

#### 1. Mean Shift

We compare the batch mean to the training mean using a **standardized z-score**:

z = | (recent_mean − training_mean) / standard_error |

Where

standard_error = training_std / sqrt(batch_size)


#### 2. Variance Shift

We compare the change in standard deviation:

std_ratio = recent_std / training_std  
std_shift = |std_ratio − 1|

### Drift Rule

Drift is flagged when:

≥ 2 features exceed drift thresholds

If drift is detected, the system warns that **model retraining may be required**.

Predictions are still returned.


## Web Application

The FastAPI application provides:

### Health Check

GET /health

### Batch Prediction

POST /predict_csv

Upload a CSV containing the required feature columns.

Example input:

lag_1,lag_2,lag_24,lag_48,rolling_mean_24,rolling_std_24,hour,day_of_week,month
0.32,0.31,0.29,1.38,0.95,0.66,1,4,2
0.30,0.32,0.30,0.33,0.95,0.66,2,4,2

The API returns:

- predictions
- drift status
- drift score
- prediction preview
- downloadable predictions CSV

## Repository Structure

ts-ml-monitoring/  
│  
│── app/&emsp;&emsp;                 FastAPI application  
│&emsp;&emsp;&emsp;├── main.py  
│&emsp;&emsp;&emsp;├── monitoring_service.py  
│&emsp;&emsp;&emsp;├── model_loader.py  
│&emsp;&emsp;&emsp;├── schemas.py  
│&emsp;&emsp;&emsp;├── templates/  
│&emsp;&emsp;&emsp;└── static/  
│  
│── src/ &emsp;&emsp;                model training and   experimentation  
│&emsp;&emsp;   ├── load_data.py  
│&emsp;&emsp;   ├── preprocess.py  
│&emsp;&emsp;   └── train_model.py  
│  
│── data/  
│&emsp;&emsp;   ├── raw/  
│&emsp;&emsp;   └── processed/  
│  
│── models/  
│  
│── Dockerfile  
│── requirements.txt  
│── README.md  

## Future Improvements

Possible extensions include:

- prediction error monitoring using delayed ground-truth labels
- automated model retraining pipelines
- statistical drift tests (KS-test, PSI)
- experiment tracking with MLflow
- streaming inference pipelines
- cloud deployment


## Author

Aryan Jha
