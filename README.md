# Crop Yield Prediction using Machine Learning
Predict crop yield using historical agricultural data with an end-to-end machine learning pipeline focusing on clean data handling, feature engineering, and model reasoning.

## Problem Statement
Accurate crop yield prediction enables farmers and policymakers to make informed decisions about food security, pricing, and resource allocation.  
This project uses historical data on rainfall, temperature, pesticide usage, and crop yields to forecast future agricultural outputs.

## Key Features
- End-to-end machine learning pipeline: Data → Features → Model → Predictions
- Modular project structure for easy maintenance and reproducibility
- Feature engineering with categorical encoding
- Regression-based yield prediction
- Model evaluation using Mean Squared Error (MSE) and R² score
- Reproducible scripts for each stage of the pipeline


## Project Structure

```plaintext
crop-yield-production/
│
├── data/
│   ├── raw/                     
│   │   ├── pesticides.csv
│   │   ├── rainfall.csv
│   │   ├── temp.csv
│   │   └── yield.csv
│   │
│   └── processed/               
│       └── final_dataset.csv
│
├── models/
│   └── crop_yield_model.pkl     
├── notebooks/                  
├── scripts/                      
│   ├── run_preprocess.py
│   ├── run_features.py
│   ├── run_train.py
│   ├── run_predict.py
│   └── test_load.py
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   ├── models/
│   │   └── train_model.py
│   │
│   └── utils/
│
├── .gitignore
└── README.md
```


## Tech Stack

**Language**
- Python 3

**Libraries**
- pandas
- numpy
- scikit-learn
- joblib

**Tools**
- VS Code
- GitHub

## Key Features
- End-to-end machine learning pipeline from data preprocessing to prediction
- Modular project structure with separate scripts for each stage
- Feature engineering using One-Hot Encoding for categorical variables
- Crop yield prediction using a Random Forest Regressor
- Model evaluation using Mean Squared Error (MSE) and R² score
- Trained model saved and reused for inference

## Machine Learning Pipeline

### 1. Data Loading
- Reads raw CSV files from `data/raw`

### 2. Preprocessing
- Data cleaning and column selection
- Merging relevant attributes

### 3. Feature Engineering
- Label encoding of categorical variables
- Selection of numerical features

### 4. Model Training
- Trains a Random Forest Regressor within a pipeline
- Splits data into training and testing sets
- Evaluation using MSE and R²
- Saves the trained model as a .pkl file

### 5. Inference
- Predicts crop yield for given inputs

## How to Run the Project

```bash
# Preprocess the data
PYTHONPATH=. python3 scripts/run_preprocess.py
# Build features
PYTHONPATH=. python3 scripts/run_features.py
# Train the model
PYTHONPATH=. python3 scripts/run_train.py
# Run prediction
PYTHONPATH=. python3 scripts/run_predict.py
```

#Sample Output
```bash
Preprocessing completed successfully
          Area   Item  Year  Yield
0  Afghanistan  Maize  1961  14000
1  Afghanistan  Maize  1962  14000
2  Afghanistan  Maize  1963  14260
3  Afghanistan  Maize  1964  14257
4  Afghanistan  Maize  1965  14400

Feature engineering completed
   Area_enc  Item_enc  Year
0         0         1  1961
1         0         1  1962
2         0         1  1963
3         0         1  1964
4         0         1  1965

Model trained successfully
MSE: 955713047.844866
R2 Score: 0.7930267194202911
Predicted Yield: 26890.18
```

## Model Reasoning
- **Random Forest Regressor** was chosen for its ability to:
  - Capture non-linear relationships
  - Handle categorical variables effectively after encoding
  - Provide robust performance with minimal feature scaling
- A **pipeline architecture** ensures consistent preprocessing during both training and inference
- Evaluation metrics (**MSE** and **R² score**) help measure prediction accuracy and model reliability

# Future Improvements
- Improve prediction accuracy by using advanced machine learning models and time-series techniques.
- Incorporate additional features such as weather and soil data for better real-world predictions.
- Deploy the model as a web-based application for easy access and real-time use.

# Author
Guttikonda Niharika
AI/ML Enthusiast

