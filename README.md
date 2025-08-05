📊 Predictive Analytics for Sales Forecasting

🧠 Overview

This project demonstrates how to build a time series sales forecasting model using Linear Regression. The goal is to predict future sales based on past sales trends, enabling businesses to make data-driven decisions about inventory, staffing, and marketing.


📌 Problem Statement

Retail businesses often struggle with fluctuating demand, leading to overstocking or stockouts. This project helps predict daily sales using historical data, providing actionable insights that improve decision-making and operational efficiency.


✅ Key Features

Clean and preprocess real-world retail sales data

Visualize sales trends over time

Create lag-based features for forecasting

Train a Linear Regression model to predict future sales

Evaluate performance using RMSE and R² score

Visualize actual vs. predicted sales to assess model accuracy


🛠️ Technologies Used

Python : Core programming language

Pandas : Data manipulation

NumPy : Numerical operations

Matplotlib : Data visualization

Scikit-learn : Machine learning model + evaluation metrics

🔁 Workflow

1. Load and preprocess data

  - Parse date columns

  - Aggregate daily sales

  - Sort chronologically

2. Feature engineering
   
  - Create lagged sales features for previous 3 days (lag_1, lag_2, lag_3)

3. Model training

  - Train/test split without shuffling (to preserve time order)

  - Fit a Linear Regression model

4. Evaluation

  - Root Mean Squared Error (RMSE)

  - R² score

Visual comparison: Actual vs. Predicted sales

📈 Results

RMSE : 512.45

R² Score : 92.30%

🎯 The model achieves over 90% R², indicating strong predictive power.
