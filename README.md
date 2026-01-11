# Predictive Maintenance for Manufacturing (Machine Learning)

## Project Overview

Unplanned machine failures can lead to costly downtime and financial loss. This project implements a Predictive Maintenance system using machine learning to anticipate machine failures before they occur, helping manufacturing units plan maintenance proactively.

## Objective

* Predict machine failure (Yes/No) using sensor data.
* Reduce unplanned downtime.
* Improve maintenance scheduling.

## Dataset

The project uses a sensor dataset (`sensor_data.csv`) containing:

| Feature          | Description                                  |
| ---------------- | -------------------------------------------- |
| timestamp        | Date and time of sensor reading              |
| vibration        | Vibration level of machine                   |
| temperature      | Operating temperature (°C)                   |
| pressure         | System pressure                              |
| rotational_speed | Machine RPM                                  |
| failure          | Target variable: 0 = No Failure, 1 = Failure |

## Technology Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost

## Workflow

1. Load dataset and inspect data.
2. Convert timestamp and sort chronologically.
3. Feature engineering: rolling mean and standard deviation for sensor readings.
4. Handle class imbalance using SMOTE.
5. Split data into train and test sets.
6. Scale features.
7. Train machine learning models (Random Forest, XGBoost).
8. Evaluate model performance.
9. Predict machine failures.

## Models & Performance

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | ~97%     |
| XGBoost       | ~99%     |

## Feature Engineering

* Rolling mean and standard deviation of vibration and temperature.
* Pressure trends for detecting anomalies.

## Evaluation Metrics

* Accuracy
* Confusion Matrix

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the model:

```
python predictive_maintenance.py
```

## Results & Insights

* XGBoost performs best among all models.
* Vibration and temperature are key indicators for failure.
* SMOTE improved model performance for minority class.

## Future Work

* Implement LSTM for deep time-series prediction.
* Build a real-time prediction dashboard.
* Deploy as a Flask or Streamlit web application.
* Add model explainability using SHAP.

## License

For educational and academic purposes.
