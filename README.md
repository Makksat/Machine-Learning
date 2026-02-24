# Machine Learning Classification Project

## Project Overview

This project builds and optimizes multiple machine learning models to predict a **binary target variable** using a structured dataset containing numerical and categorical features.

The project covers the full machine learning pipeline:

- Dataset Exploration  
- Exploratory Data Analysis (EDA)  
- Data Preprocessing  
- Feature Engineering  
- Model Training & Evaluation  
- Cross-Validation  
- Hyperparameter Tuning  
- Model Interpretation (Feature Importance & SHAP)  
- Model Saving  


##  Dataset Description

Dataset size:
- **9000 rows**
- **11 columns**
- 8 numerical features  
- 2 categorical features  
- 1 binary target  

### Features

- `feature_1` to `feature_8` (Numerical)
- `category_1`, `category_2` (Categorical)
- `target` (Binary: 0 or 1)

Key findings:
- 900 missing values handled during preprocessing
- No duplicate rows
- Balanced binary target


## Exploratory Data Analysis (EDA)

### Distribution Analysis
- Histograms for numerical features
- Boxplots for outlier detection
- Heatmap for correlation analysis
- Scatterplots and pairplots

### Statistical Testing
- **Chi-Square Test**
  - `category_1` strongly related to target
  - Region variables showed weak significance
- **T-Test**
  - Significant differences found across several numerical features

EDA helped identify important predictors and irrelevant features.


## Data Preprocessing

### Missing Values
- Replaced missing values in `feature_3` and `feature_6` using **median**

### Outlier Handling
- Used **Interquartile Range (IQR)**
- Clipped extreme values instead of removing them

### Encoding
- `category_1` → Label Encoding
- `category_2` → One-Hot Encoding

### Scaling
- Applied **StandardScaler** to numerical features


## ⚙ Feature Engineering

Created additional features to improve model performance:

- Multiplicative interactions:
  - `f1_f2 = feature_1 * feature_2`
  - `f3_f4 = feature_3 * feature_4`
- Additive interactions:
  - `f1_f2_sum`
  - `f2_f4_sum`
- Polynomial feature:
  - `f4_squared`
- Aggregated features:
  - `total`, `avg`, `max`, `min`

Dropped irrelevant or low-significance features based on statistical tests.


## Models Implemented

The following models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  
- ROC Curves  


## Cross-Validation

- Used **5-Fold Cross-Validation**
- Evaluated:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1-score (weighted)
  - ROC-AUC

Ensured model performance was consistent and robust.


## Hyperparameter Tuning

Used **GridSearchCV** with 5-fold cross-validation for:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost

### Best Model

**Gradient Boosting achieved the highest performance after tuning.**


