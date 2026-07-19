# ✈️ Airline Delay Intelligence Dashboard

An interactive data science project for exploring U.S. airline delay patterns and predicting flight outcomes using XGBoost models.

## Project Overview

This project combines exploratory data analysis, machine learning, and an interactive Streamlit dashboard to:

- Explore delay patterns by airline, airport, and month.
- Compare departure delay with arrival delay.
- Classify flights as **On Time** or **Delayed**.
- Estimate expected arrival delay in minutes.
- Present results through a polished interactive dashboard.

## Dashboard Features

### Overview
- Total number of flights.
- Number of airlines.
- Average arrival delay.
- Percentage of delayed flights.
- Filterable flight records.

### Exploratory Data Analysis
- Flights by airline.
- Average arrival delay by airline.
- Average arrival delay by month.
- Departure delay vs. arrival delay.
- Top origin airports by average delay.
- Delayed vs. on-time flights.
- Correlation heatmap.

### Machine Learning

#### Classification
The classification model predicts whether a flight is likely to be delayed by at least 15 minutes.

Model inputs:
- Flight date.
- Airline.
- Scheduled departure time.
- Scheduled flight duration.
- Flight distance.

Evaluation:
- Accuracy.
- Classification report.
- Confusion matrix.
- ROC curve and AUC.

#### Regression
The regression model estimates arrival delay in minutes.

Model inputs:
- Departure delay.
- Flight distance.
- Month.
- Day.
- Day of week.

Evaluation:
- RMSE.
- MAE.
- R² score.
- Actual vs. predicted plot.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

## Repository Structure

```text
airline-delay-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── small_clean_data_v2.csv
│
├── models/
│   ├── xgb_classification_model.pkl
│   └── xgb_regression_model.pkl
│
└── notebooks/
    └── MY_EDA_PROJECT.ipynb
```

## Run the Project Locally

1. Clone the repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## Dataset

The project uses the U.S. Department of Transportation flight delay dataset available on Kaggle. Cancelled and diverted flights were removed, missing values were handled, and airline and airport reference tables were merged into the main dataset.

A balanced monthly sample was created for the classification model to reduce memory usage while maintaining representation across all months.

## Important Note

Predictions are most reliable when the entered values remain within realistic ranges similar to the training data. The dashboard restricts numeric inputs to practical values to reduce unrealistic predictions.

## Author

Developed by **Turki** as a data science and artificial intelligence project.
