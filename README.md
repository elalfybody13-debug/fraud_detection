# Credit Card Fraud Detection

Machine Learning project for detecting fraudulent credit card transactions.

## Project Overview

This project uses Machine Learning to classify credit card transactions as:

- Normal
- Fraudulent

The project focuses on handling highly imbalanced data and evaluating the model using appropriate classification metrics.

## Machine Learning

The main model used in the project is:

- XGBoost

Other models tested:

- Logistic Regression
- Random Forest

## Features

The model uses features such as:

- Transaction amount
- Transaction category
- Gender
- State
- Customer location
- Merchant location
- Distance between customer and merchant
- Customer age
- Transaction hour
- Day of week
- Month
- Weekend indicator

## Preprocessing

The project uses:

- StandardScaler
- OneHotEncoder
- Feature Engineering
- Class imbalance handling

## Evaluation Metrics

The models are evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

## Streamlit Application

The project includes a Streamlit application for testing transactions and predicting fraud probability.

