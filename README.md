# ✈️ Airline Delay Intelligence Dashboard

An interactive data science project that explores U.S. airline delay patterns and predicts flight delays using machine learning.

The project combines **Exploratory Data Analysis (EDA)**, **XGBoost models**, and an interactive **Streamlit dashboard** to better understand airline performance and flight delays.

---

## 🌐 Live Demo

[STREAMLIT](https://airline-delay-dashboard-uzckqchivbogtd5wsghykr.streamlit.app/)
---

## 🎯 Project Objectives

This project covers the complete data science workflow:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Interactive visualizations
- Flight delay prediction using Machine Learning
- Streamlit dashboard deployment

---

## 📂 Dataset

The project uses the US DOT Flight Delays dataset containing millions of domestic flight records.

| Dataset Stage | Rows | Columns |
|---|---:|---:|
| Original Dataset | 5,819,079 | 31 |
| Cleaned Dataset | 5,714,008 | 32 |
| Model Sample | 96,000 | 33 |

The original dataset contains more than 5.8 million U.S. domestic flight records. After removing cancelled and diverted flights, handling missing values, and merging airline and airport information, the cleaned dataset contained 5,714,008 records.

To reduce training time while preserving seasonal representation, a balanced sample of 96,000 flights was created by selecting 8,000 flights from each month.

---

## 📊 Exploratory Data Analysis

The notebook includes visualizations such as:

- Flights by airline
- Average delay by airline
- Monthly delay trends
- Departure vs Arrival delay
- Top delayed airports
- Delay distribution
- Correlation heatmap

Each chart includes a short explanation of the findings.

---

## 🤖 Machine Learning

Two XGBoost models were developed.

### Flight Delay Classification

Predicts whether a flight will arrive **15 minutes or more late**.

**Features**

- Flight date
- Airline
- Scheduled departure time
- Scheduled flight duration
- Flight distance

---

### Arrival Delay Regression

Predicts the expected arrival delay in minutes.

**Features**

- Departure delay
- Flight distance
- Month
- Day
- Day of week

---

## 💻 Streamlit Dashboard

The dashboard allows users to:

- Filter flights by month and airline
- Explore interactive charts
- Predict flight status
- Predict arrival delay
- View prediction confidence

---

## 📁 Repository Structure

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

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/torki112231/airline-delay-dashboard.git
```

2. Install the required packages

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
streamlit run app.py
```

---

## 🛠️ Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Google Colab

---

## 🔍 Key Findings

- Flight delays vary noticeably across airlines.
- Departure delay is the strongest indicator of arrival delay.
- Delay patterns change throughout the year.
- Machine learning can provide useful predictions using scheduled flight information.

---

## 👨‍💻 Author

**Turki AbuHAimed**

Data Science & Artificial Intelligence Bootcamp
