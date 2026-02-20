# 🚀 Spaceship Titanic — Passenger Transport Prediction

A machine learning project that predicts whether passengers aboard the Spaceship Titanic were transported to an alternate dimension, using feature engineering and Gradient Boosting.

## Overview

Based on the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition, this project performs end-to-end ML — from exploratory data analysis through feature engineering to model selection with hyperparameter tuning.

## Tech Stack

- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn (RandomForest, GradientBoosting, GridSearchCV)

## Approach

### 1. Exploratory Data Analysis
- Analyzed distributions of Age, spending columns (RoomService, FoodCourt, etc.)
- Discovered **CryoSleep** as a strong predictor (~82% transport rate vs ~33%)
- Cross-tabulated HomePlanet vs Destination travel patterns
- Examined VIP vs non-VIP transport rates

### 2. Feature Engineering
- Split `Cabin` (e.g., "B/1/S") into **Deck**, **CabinNum**, **Side**
- Created `TotalSpend` = sum of all spending columns
- Added `_missing` indicator flags for spending columns with nulls
- Encoded categoricals with LabelEncoder

### 3. Data Preprocessing
- Imputed missing values: median (Age), mode (CryoSleep, VIP), "unknown" (categoricals), 0 (spending)
- Dropped non-predictive columns: Name, PassengerId, CabinNum

### 4. Model Training & Selection
- Compared **RandomForest** vs **GradientBoosting** via GridSearchCV (5-fold CV, ROC-AUC scoring)
- Selected **GradientBoostingClassifier** with tuned hyperparameters

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~79-81% |
| ROC-AUC | Optimized via GridSearchCV |
| Precision/Recall | Balanced across both classes |

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook "SPACESHIP PROJECT_commented.ipynb"
```

## Project Structure

```
├── SPACESHIP PROJECT_commented.ipynb   # Full pipeline (EDA → Model), fully commented
├── train.csv                           # Training dataset
└── README.md
```

## Key Insights

- **CryoSleep** is the strongest single predictor of being transported
- VIP passengers were *less* likely to be transported
- Spending features are heavily right-skewed — most passengers spent 0
- Gradient Boosting outperformed Random Forest after hyperparameter tuning
