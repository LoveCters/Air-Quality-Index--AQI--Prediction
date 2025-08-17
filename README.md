# Air Quality Index (AQI) Prediction

## Overview

This project predicts the **Air Quality Index (AQI)** based on historical air quality data using both **machine learning** and **deep learning** techniques. It includes data preprocessing, feature scaling, model training, and performance evaluation.

The main goal is to explore and compare different modeling approaches for accurate AQI forecasting.

---

## Features

* Data loading and preprocessing from CSV.
* Exploratory Data Analysis (EDA) with visualizations.
* Implementation of:

  * **Linear Regression** (Baseline model)
  * **Deep Learning (LSTM/ANN)** models using TensorFlow/Keras.
* Model evaluation with metrics such as **MAE**, **MSE**, and **R²**.
* Prediction visualization for performance comparison.

---

## Dataset

* **File**: `AirQualityIndex6years.csv`
* Contains multi-year air quality data with features like:

  * Pollutants: O₃, CO, NO₂, SO₂, PM2.5
  * Weather: Temperature, Humidity
  * AQI values

---

## Technologies Used

* **Python 3.x**
* **Pandas**, **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **TensorFlow / Keras**

---

## Model Workflow

1. **Data Loading** → Read CSV into DataFrame.
2. **Preprocessing** → Handle missing values, scale features.
3. **Modeling** → Train baseline and deep learning models.
4. **Evaluation** → Compare performance across models.
5. **Prediction** → Forecast AQI for unseen data.





