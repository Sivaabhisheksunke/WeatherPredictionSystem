**🌦️ Weather Prediction – Machine Learning Project**
**📌 Project Overview**
This project builds a Machine Learning model to predict weather conditions using historical meteorological data. The dataset was sourced from Kaggle and contains atmospheric features such as temperature, humidity, wind speed, pressure, and visibility.
The goal of this project is to analyze weather patterns and develop classification models capable of predicting standardized weather categories such as:
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
The original dataset contained 50 unique weather labels (e.g., "Thunderstorms, Moderate Rain Showers, Fog").
To improve model performance and reduce complexity:
Custom text parsing was applied
Weather conditions were standardized into 4 major categories:
Clear
Cloudy
Rain
Snow
This significantly simplified the classification problem.

**🛠️ Technologies & Libraries Used**
Python
Pandas & NumPy (Data manipulation)
Matplotlib & Seaborn (Visualization)
Scikit-learn (Machine Learning)
KaggleHub (Dataset retrieval)
Google Colab (Development environment)

**🔍 Exploratory Data Analysis (EDA)**
Checked data types and missing values
Correlation heatmap analysis
Distribution plots (histograms & boxplots)
Class balance analysis
Removed duplicates
Feature scaling using StandardScaler

**🤖 Machine Learning Models Implemented**
The following classification algorithms were trained and compared:
Decision Tree
Random Forest
Logistic Regression
Support Vector Machine (SVM)
Naive Bayes
K-Nearest Neighbors (KNN)

**📈 Model Evaluation**
Evaluation metrics used:
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
5-Fold Cross Validation

**🔥 Best Performing Model: Random Forest**
Test Accuracy: ~69%
Cross-Validation Accuracy: ~62%
Hyperparameter tuning performed using GridSearchCV
Best parameters found:
n_estimators = 100
max_features = 'sqrt'

**🚀 Key Learnings & Skills Demonstrated**
✔ Data Cleaning & Preprocessing
✔ Feature Engineering (Text Parsing & Standardization)
✔ Exploratory Data Analysis
✔ Handling Class Imbalance
✔ Model Comparison & Evaluation
✔ Cross-Validation
✔ Hyperparameter Tuning
✔ End-to-End ML Pipeline Development

**📌 Future Improvements**
Try XGBoost / Gradient Boosting
Use time-series based weather forecasting models
Perform advanced feature engineering
Improve accuracy using ensemble stacking
Deploy as a Flask / Streamlit web app

**💡 Conclusion**
This project showcases how raw weather data can be transformed into a structured machine learning pipeline capable of predicting atmospheric conditions.
It demonstrates strong foundations in data preprocessing, exploratory analysis, classification modeling, and performance optimization.
