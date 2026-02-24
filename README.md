**🌦️ Weather Prediction – Machine Learning Project**
**📌 Project Overview**

This project builds a Machine Learning model to predict weather conditions using historical meteorological data.

The dataset was sourced from Kaggle and contains atmospheric features such as temperature, humidity, wind speed, pressure, and visibility.

The goal of this project is to analyze weather patterns and develop classification models capable of predicting standardized weather categories:

Clear

Cloudy

Rain

Snow

This project demonstrates the complete machine learning lifecycle — from raw data preprocessing to model optimization and evaluation.

**📊 Dataset Information**

Source: Kaggle
Records: 8,784 entries

Features:

Temperature (°C)

Dew Point Temperature

Relative Humidity (%)

Wind Speed (km/h)

Visibility (km)

Atmospheric Pressure (kPa)

Weather Description

**🔎 Feature Engineering**

The original dataset contained 50 unique weather labels
(e.g., "Thunderstorms, Moderate Rain Showers, Fog").

To improve model performance and reduce complexity:

Applied custom text parsing

Standardized weather conditions into 4 major categories:

Clear

Cloudy

Rain

Snow

This significantly simplified the classification problem.

**🛠️ Technologies & Libraries Used**

Python

Pandas & NumPy

Matplotlib & Seaborn

Scikit-learn

KaggleHub

Google Colab

**🔍 Exploratory Data Analysis (EDA)**

Checked data types and missing values

Correlation heatmap analysis

Distribution plots (histograms & boxplots)

Class balance analysis

Removed duplicates

Feature scaling using StandardScaler

**🤖 Machine Learning Models Implemented**

Decision Tree

Random Forest

Logistic Regression

Support Vector Machine (SVM)

Naive Bayes

K-Nearest Neighbors (KNN)

**📈 Model Evaluation**

Metrics Used:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

5-Fold Cross Validation

**🔥 Best Performing Model: Random Forest**

Test Accuracy: ~69%

Cross-Validation Accuracy: ~62%

Hyperparameter tuning performed using GridSearchCV

Best Parameters:

n_estimators = 100

max_features = 'sqrt'

**🚀 Key Skills Demonstrated**

Data Cleaning & Preprocessing

Feature Engineering

Exploratory Data Analysis

Model Comparison

Cross-Validation

Hyperparameter Tuning

End-to-End ML Pipeline Development

**📌 Future Improvements**

Implement XGBoost / Gradient Boosting

Use time-series forecasting models

Perform advanced feature engineering

Improve accuracy using ensemble stacking

Deploy as a Flask / Streamlit web app

**💡 Conclusion**

This project demonstrates how raw weather data can be transformed into a structured machine learning pipeline capable of predicting atmospheric conditions.

It highlights strong skills in preprocessing, exploratory analysis, model building, and performance optimization.
