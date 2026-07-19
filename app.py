import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import date
from html import escape


# PAGE CONFIGURATION

st.set_page_config(
    page_title='Airline Delay Dashboard',
    page_icon='✈️',
    layout='wide'
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --panel: rgba(13, 30, 49, 0.88);
        --panel-2: rgba(16, 39, 62, 0.82);
        --border: rgba(125, 211, 252, 0.17);
        --text: #ecf8ff;
        --muted: #9db4c8;
        --cyan: #22d3ee;
        --blue: #3b82f6;
        --amber: #f59e0b;
        --red: #fb7185;
        --green: #34d399;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(34, 211, 238, 0.13), transparent 26%),
            radial-gradient(circle at 15% 25%, rgba(59, 130, 246, 0.12), transparent 28%),
            linear-gradient(145deg, #050b14 0%, #081525 48%, #07111f 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(8, 23, 39, 0.98), rgba(5, 14, 25, 0.98));
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.6rem;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.6rem 2.7rem;
        margin-bottom: 1.3rem;
        border: 1px solid rgba(125, 211, 252, 0.22);
        border-radius: 28px;
        background:
            linear-gradient(120deg, rgba(12, 33, 54, 0.97), rgba(12, 52, 73, 0.84)),
            radial-gradient(circle at 80% 20%, rgba(34, 211, 238, 0.2), transparent 35%);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
    }

    .hero::after {
        content: '✈';
        position: absolute;
        right: 4%;
        top: 8%;
        font-size: 9rem;
        line-height: 1;
        color: rgba(125, 211, 252, 0.10);
        transform: rotate(-12deg);
    }

    .eyebrow {
        color: #67e8f9;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .hero h1 {
        margin: 0;
        color: #f4fbff;
        font-size: clamp(2.3rem, 5vw, 4.4rem);
        line-height: 1.02;
        letter-spacing: -0.05em;
        max-width: 850px;
    }

    .hero p {
        color: #b9cedd;
        font-size: 1.08rem;
        line-height: 1.75;
        max-width: 800px;
        margin: 1.1rem 0 0;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.5rem;
    }

    .hero-pill {
        border: 1px solid rgba(103, 232, 249, 0.24);
        background: rgba(8, 31, 49, 0.72);
        color: #d9f7ff;
        border-radius: 999px;
        padding: 0.55rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .sidebar-brand {
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.15rem;
        margin-bottom: 1.3rem;
        background: linear-gradient(145deg, rgba(17, 46, 70, 0.9), rgba(9, 27, 43, 0.88));
    }

    .sidebar-brand-title {
        color: #e8fbff;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .sidebar-brand-copy {
        color: #93adbf;
        font-size: 0.84rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(13, 35, 55, 0.95), rgba(9, 25, 41, 0.95));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.25rem 1.3rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
        min-height: 135px;
    }

    div[data-testid="stMetric"] label {
        color: #8eb0c5 !important;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    div[data-testid="stMetricValue"] {
        color: #f0fbff;
        font-weight: 800;
        letter-spacing: -0.035em;
    }

    .section-title {
        margin: 1.25rem 0 0.9rem;
        color: #f0fbff;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.035em;
    }

    .section-copy {
        color: var(--muted);
        margin-top: -0.45rem;
        margin-bottom: 1.25rem;
        line-height: 1.7;
    }

    .result-card {
        border-radius: 24px;
        padding: 1.45rem 1.55rem;
        margin-top: 1rem;
        border: 1px solid;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.24);
    }

    .result-delayed {
        border-color: rgba(251, 113, 133, 0.36);
        background: linear-gradient(145deg, rgba(80, 25, 40, 0.88), rgba(40, 19, 31, 0.92));
    }

    .result-ontime {
        border-color: rgba(52, 211, 153, 0.34);
        background: linear-gradient(145deg, rgba(14, 70, 58, 0.82), rgba(10, 42, 38, 0.94));
    }

    .result-info {
        border-color: rgba(34, 211, 238, 0.34);
        background: linear-gradient(145deg, rgba(11, 57, 77, 0.86), rgba(8, 35, 52, 0.94));
    }

    .result-kicker {
        color: #a8c2d3;
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.18em;
    }

    .result-main {
        margin-top: 0.45rem;
        color: #f2fcff;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .result-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1.15rem;
    }

    .result-stat {
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(3, 13, 24, 0.32);
        border-radius: 16px;
        padding: 0.9rem 1rem;
    }

    .result-stat-label {
        color: #91aabd;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .result-stat-value {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 0.15rem;
    }

    div[data-testid="stTabs"] button {
        border-radius: 12px 12px 0 0;
        padding-left: 1.05rem;
        padding-right: 1.05rem;
        font-weight: 700;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, var(--cyan), var(--blue));
        height: 3px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stDateInput"] > div > div,
    [data-testid="stTimeInput"] > div > div {
        background: rgba(10, 27, 44, 0.86) !important;
        border-color: rgba(125, 211, 252, 0.18) !important;
        border-radius: 13px !important;
    }

    div[data-testid="stSlider"] {
        padding: 0.5rem 0.2rem 0.8rem;
    }

    div[data-testid="stButton"] button {
        border: 0;
        border-radius: 14px;
        background: linear-gradient(100deg, #06b6d4, #2563eb);
        color: white;
        font-weight: 800;
        min-height: 3rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
        transition: transform .18s ease, box-shadow .18s ease;
    }

    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 35px rgba(6, 182, 212, 0.28);
        color: white;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 18px;
        overflow: hidden;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stImage"],
    [data-testid="stPyplotGlobalUse"] {
        border-radius: 20px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(125, 211, 252, 0.13);
    }

    @media (max-width: 800px) {
        .hero { padding: 1.7rem 1.4rem; }
        .hero::after { font-size: 5.5rem; right: -2%; }
        .result-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

sns.set_style('whitegrid')


# LOAD DATA AND MODELS

@st.cache_data
def load_data():
    return pd.read_csv(
        'data/small_clean_data_v2.csv')


@st.cache_resource
def load_models():
    classification_model = joblib.load(
        'models/xgb_classification_model.pkl')

    regression_model = joblib.load(
        'models/xgb_regression_model.pkl')

    return classification_model, regression_model


df = load_data()

model_classification, model_regression = load_models()

# MONTH NAMES

month_names = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}


# REALISTIC INPUT LIMITS BASED ON THE DATASET

scheduled_time_min = max(
    20,
    int(df['SCHEDULED_TIME'].quantile(0.01))
)

scheduled_time_max = min(
    1000,
    int(df['SCHEDULED_TIME'].quantile(0.99))
)

distance_min = max(
    30,
    int(df['DISTANCE'].quantile(0.01))
)

distance_max = min(
    6000,
    int(df['DISTANCE'].quantile(0.99))
)


departure_delay_min = max(
    -60,
    int(df['DEPARTURE_DELAY'].quantile(0.01))
)

departure_delay_max = min(
    600,
    int(df['DEPARTURE_DELAY'].quantile(0.99))
)


# DASHBOARD HERO

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">US Flight Intelligence • XGBoost Analytics</div>
        <h1>Airline Delay Intelligence Dashboard</h1>
        <p>
            Explore delay patterns across airlines and airports, then use machine-learning
            models to estimate flight status and expected arrival delay.
        </p>
        <div class="hero-pills">
            <span class="hero-pill">✈️ Flight analytics</span>
            <span class="hero-pill">📊 Interactive EDA</span>
            <span class="hero-pill">🤖 Classification</span>
            <span class="hero-pill">⏱️ Delay regression</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)


# SIDEBAR FILTERS

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">✈️ FlightScope AI</div>
        <div class="sidebar-brand-copy">
            Filter the dashboard, explore flight performance, and test the prediction models.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header('🔎 FILTERS')

month_options = ['All'] + sorted(df['MONTH'].dropna().unique().tolist())

selected_month = st.sidebar.selectbox(
    'Select Month',
    options=month_options,
    format_func=lambda x: 'All' if x == 'All' else month_names.get(int(x), str(x))
)

if 'AIRLINE_NAME' in df.columns:
    airline_options = ['All'] + sorted(
        df['AIRLINE_NAME'].dropna().unique().tolist()
    )
else:
    airline_options = ['All'] + sorted(
        df['AIRLINE'].dropna().unique().tolist()
    )

selected_airline = st.sidebar.selectbox(
    'Select Airline',
    options=airline_options
)


# FILTER DATA

filtered_df = df.copy()

if selected_month != 'All':
    filtered_df = filtered_df[
        filtered_df['MONTH'] == selected_month
    ]

airline_filter_column = (
    'AIRLINE_NAME'
    if 'AIRLINE_NAME' in filtered_df.columns
    else 'AIRLINE'
)

if selected_airline != 'All':
    filtered_df = filtered_df[
        filtered_df[airline_filter_column] == selected_airline
    ]


# CREATE TABS

tab1, tab2, tab3 = st.tabs([
    '🏠 Overview',
    '📈 Explore Delays',
    '🤖 Predict'
])


# SUMMARY TAB

with tab1:

    st.markdown('<div class="section-title">Flight Performance Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A quick snapshot of the flights matching your current filters.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        'TOTAL FLIGHTS',
        f'{len(filtered_df):,}'
    )

    col2.metric(
        'AIRLINES',
        filtered_df[airline_filter_column].nunique()
    )

    average_arrival_delay = (
        filtered_df['ARRIVAL_DELAY'].mean()
        if len(filtered_df) > 0
        else 0
    )

    col3.metric(
        'AVERAGE ARRIVAL DELAY',
        f'{average_arrival_delay:.2f} min'
    )

    delayed_percentage = (
        filtered_df['IS_DELAYED'].mean() * 100
        if 'IS_DELAYED' in filtered_df.columns and len(filtered_df) > 0
        else 0
    )

    col4.metric(
        'DELAYED FLIGHTS',
        f'{delayed_percentage:.2f}%'
    )

    st.markdown('<div class="section-title">Filtered Flight Records</div>', unsafe_allow_html=True)

    preview_columns = [
        column for column in [
            'AIRLINE_NAME',
            'AIRLINE',
            'ORIGIN_AIRPORT_NAME',
            'ORIGIN_AIRPORT',
            'DESTINATION_AIRPORT',
            'MONTH',
            'DAY',
            'DEPARTURE_DELAY',
            'ARRIVAL_DELAY',
            'IS_DELAYED'
        ]
        if column in filtered_df.columns
    ]

    preview_df = filtered_df[preview_columns].copy()

    if 'MONTH' in preview_df.columns:
        preview_df['MONTH'] = (
            preview_df['MONTH']
            .map(month_names)
            .fillna(preview_df['MONTH'])
        )

    st.dataframe(
        preview_df.head(20),
        use_container_width=True
    )


# EDA CHARTS TAB

with tab2:

    st.markdown('<div class="section-title">Explore Delay Patterns</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Select a visualization to investigate airline, airport, seasonal, and operational patterns.</div>',
        unsafe_allow_html=True
    )

    selected_chart = st.selectbox(
        'Select Chart',
        [
            'Flights by Airline',
            'Average Arrival Delay by Airline',
            'Average Arrival Delay by Month',
            'Departure Delay vs Arrival Delay',
            'Top 10 Origin Airports',
            'Delayed vs On-Time Flights',
            'Correlation Heatmap'
        ]
    )

    if filtered_df.empty:
        st.warning('No data is available for the selected filters.')

    elif selected_chart == 'Flights by Airline':

        plt.figure(figsize=(10, 7))

        order = (
            filtered_df[airline_filter_column]
            .value_counts()
            .index
        )

        sns.countplot(
            data=filtered_df,
            y=airline_filter_column,
            order=order,
            hue=airline_filter_column,
            palette='pastel',
            legend=False
        )

        plt.title('Flights by Airline')
        plt.xlabel('Number of Flights')
        plt.ylabel('Airline')

        st.pyplot(plt.gcf())
        plt.close()


    elif selected_chart == 'Average Arrival Delay by Airline':

        plt.figure(figsize=(10, 7))

        airline_delay = (
            filtered_df
            .groupby(airline_filter_column)['ARRIVAL_DELAY']
            .mean()
            .sort_values(ascending=False)
        )

        sns.barplot(
            x=airline_delay.values,
            y=airline_delay.index,
            hue=airline_delay.index,
            palette='pastel',
            legend=False
        )

        plt.title('Average Arrival Delay by Airline')
        plt.xlabel('Average Arrival Delay (Minutes)')
        plt.ylabel('Airline')

        st.pyplot(plt.gcf())
        plt.close()


    elif selected_chart == 'Average Arrival Delay by Month':

        monthly_delay = (
            filtered_df.groupby('MONTH')['ARRIVAL_DELAY']
            .mean()
            .reset_index()
            .sort_values('MONTH')
        )

        monthly_delay['MONTH_NAME'] = (
            monthly_delay['MONTH'].map(month_names)
        )

        plt.figure(figsize=(12, 6))

        sns.barplot(
            data=monthly_delay,
            x='MONTH_NAME',
            y='ARRIVAL_DELAY',
            hue='MONTH_NAME',
            palette='pastel',
            legend=False
        )

        plt.title('Average Arrival Delay by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Arrival Delay (Minutes)')
        plt.xticks(rotation=45)

        st.pyplot(plt.gcf())
        plt.close()


    elif selected_chart == 'Departure Delay vs Arrival Delay':

        scatter_df = filtered_df[
            ['DEPARTURE_DELAY', 'ARRIVAL_DELAY']
        ].dropna()

        if len(scatter_df) > 10000:
            scatter_df = scatter_df.sample(
                10000,
                random_state=42
            )

        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=scatter_df,
            x='DEPARTURE_DELAY',
            y='ARRIVAL_DELAY',
            alpha=0.4
        )

        plt.title('Departure Delay vs Arrival Delay')
        plt.xlabel('Departure Delay (Minutes)')
        plt.ylabel('Arrival Delay (Minutes)')

        st.pyplot(plt.gcf())
        plt.close()


    elif selected_chart == 'Top 10 Origin Airports':

        airport_column = (
            'ORIGIN_AIRPORT_NAME'
            if 'ORIGIN_AIRPORT_NAME' in filtered_df.columns
            else 'ORIGIN_AIRPORT'
        )

        top_airports = (
            filtered_df
            .groupby(airport_column)['ARRIVAL_DELAY']
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        plt.figure(figsize=(10, 7))

        sns.barplot(
            x=top_airports.values,
            y=top_airports.index,
            hue=top_airports.index,
            palette='pastel',
            legend=False
        )

        plt.title('Top 10 Origin Airports with Highest Average Delay')
        plt.xlabel('Average Arrival Delay (Minutes)')
        plt.ylabel('Origin Airport')

        st.pyplot(plt.gcf())
        plt.close()


    elif selected_chart == 'Delayed vs On-Time Flights':

        delay_counts = (
            filtered_df['IS_DELAYED']
            .value_counts()
            .sort_index()
        )

        delay_labels = [
            'On Time' if value == 0 else 'Delayed'
            for value in delay_counts.index
        ]

        plt.figure(figsize=(6, 5))

        sns.barplot(
            x=delay_labels,
            y=delay_counts.values,
            hue=delay_labels,
            palette='pastel',
            legend=False
        )

        plt.title('Delayed vs On-Time Flights')
        plt.xlabel('Flight Status')
        plt.ylabel('Number of Flights')

        st.pyplot(plt.gcf())
        plt.close()


    elif selected_chart == 'Correlation Heatmap':

        correlation_columns = [
            column for column in [
                'DEPARTURE_DELAY',
                'ARRIVAL_DELAY',
                'AIR_TIME',
                'ELAPSED_TIME',
                'DISTANCE',
                'TAXI_OUT',
                'TAXI_IN'
            ]
            if column in filtered_df.columns
        ]

        plt.figure(figsize=(9, 7))

        sns.heatmap(
            filtered_df[correlation_columns].corr(),
            annot=True,
            fmt='.2f',
            cmap='coolwarm'
        )

        plt.title('Correlation Heatmap')

        st.pyplot(plt.gcf())
        plt.close()


# PREDICTION TAB

with tab3:

    st.markdown('<div class="section-title">Flight Delay Prediction Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Choose a prediction task and enter realistic flight information.</div>',
        unsafe_allow_html=True
    )

    prediction_type = st.radio(
        'What would you like to predict?',
        ['Flight Status', 'Arrival Delay'],
        horizontal=True,
        help=(
            'Flight Status predicts whether the flight is likely to be delayed. '
            'Arrival Delay predicts the expected delay in minutes.'
        )
    )

    if prediction_type == 'Flight Status':

        st.write(
            'Enter the planned flight details. '
            'The model will predict whether the flight is likely to arrive late.'
        )

        col1, col2 = st.columns(2)

        with col1:
            flight_date = st.date_input(
                'Flight Date',
                min_value=date.today(),
                value=date.today(),
                help='Choose the planned date of the flight.'
            )

            airline = st.selectbox(
                'Airline',
                sorted(df['AIRLINE_NAME'].dropna().unique()),
                help='Choose the airline operating the flight.'
            )

            departure_time = st.time_input(
                'Scheduled Departure Time',
                help='Choose the planned departure time.'
            )

        with col2:
            scheduled_time = st.slider(
                'Scheduled Flight Duration (minutes)',
                min_value=scheduled_time_min,
                max_value=scheduled_time_max,
                value=min(max(120, scheduled_time_min), scheduled_time_max),
                step=5,
                help=(
                    f'Choose a realistic planned duration between '
                    f'{scheduled_time_min} and {scheduled_time_max} minutes.'
                )
            )

            distance = st.slider(
                'Flight Distance (miles)',
                min_value=distance_min,
                max_value=distance_max,
                value=min(max(500, distance_min), distance_max),
                step=10,
                help=(
                    f'Choose a realistic distance between '
                    f'{distance_min} and {distance_max} miles.'
                )
            )

        month = flight_date.month
        day = flight_date.day
        day_of_week = flight_date.isoweekday()

        scheduled_departure = (
            departure_time.hour * 100
            + departure_time.minute
        )

        if st.button(
            'Predict Flight Status',
            type='primary',
            use_container_width=True
        ):

            model_features = (
                model_classification
                .get_booster()
                .feature_names
            )

            classification_input = pd.DataFrame(
                0,
                index=[0],
                columns=model_features
            )

            classification_input.loc[0, 'MONTH'] = month
            classification_input.loc[0, 'DAY'] = day
            classification_input.loc[0, 'DAY_OF_WEEK'] = day_of_week
            classification_input.loc[0, 'SCHEDULED_DEPARTURE'] = scheduled_departure
            classification_input.loc[0, 'SCHEDULED_TIME'] = scheduled_time
            classification_input.loc[0, 'DISTANCE'] = distance

            airline_column = f'AIRLINE_NAME_{airline}'

            if airline_column in classification_input.columns:
                classification_input.loc[0, airline_column] = 1

            prediction = model_classification.predict(
                classification_input
            )[0]

            delay_probability = model_classification.predict_proba(
                classification_input
            )[0][1] * 100

            status_text = 'Likely Delayed' if prediction == 1 else 'Likely On Time'
            card_class = 'result-delayed' if prediction == 1 else 'result-ontime'
            status_icon = '⚠️' if prediction == 1 else '✅'

            confidence = (
                'High'
                if delay_probability >= 75 or delay_probability <= 25
                else 'Moderate'
                if delay_probability >= 60 or delay_probability <= 40
                else 'Low'
            )

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="result-kicker">Prediction result</div>
                    <div class="result-main">{status_icon} {escape(status_text)}</div>
                    <div class="result-grid">
                        <div class="result-stat">
                            <div class="result-stat-label">Chance of delay</div>
                            <div class="result-stat-value">{delay_probability:.1f}%</div>
                        </div>
                        <div class="result-stat">
                            <div class="result-stat-label">Model confidence</div>
                            <div class="result-stat-value">{confidence}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.write(
            'Enter the actual departure information. '
            'The model will estimate the arrival delay in minutes.'
        )

        col1, col2 = st.columns(2)

        with col1:
            flight_date = st.date_input(
                'Flight Date',
                min_value=date.today(),
                value=date.today(),
                key='regression_flight_date',
                help='Choose the date of the flight.'
            )

            distance = st.slider(
                'Flight Distance (miles)',
                min_value=distance_min,
                max_value=distance_max,
                value=min(max(500, distance_min), distance_max),
                step=10,
                key='regression_distance',
                help=(
                    f'Choose a realistic distance between '
                    f'{distance_min} and {distance_max} miles.'
                )
            )

        with col2:
            departure_delay = st.slider(
                'Departure Delay (minutes)',
                min_value=departure_delay_min,
                max_value=departure_delay_max,
                value=0,
                step=5,
                help=(
                    f'Choose a value between {departure_delay_min} and '
                    f'{departure_delay_max} minutes. Use 0 when the flight '
                    f'departed on time; negative values mean it departed early.'
                )
            )

        month = flight_date.month
        day = flight_date.day
        day_of_week = flight_date.isoweekday()

        if st.button(
            'Predict Arrival Delay',
            type='primary',
            use_container_width=True
        ):

            regression_input = pd.DataFrame(
                [[
                    departure_delay,
                    distance,
                    month,
                    day,
                    day_of_week
                ]],
                columns=[
                    'DEPARTURE_DELAY',
                    'DISTANCE',
                    'MONTH',
                    'DAY',
                    'DAY_OF_WEEK'
                ]
            )

            predicted_delay = model_regression.predict(
                regression_input
            )[0]

            if predicted_delay > 0:
                result_title = 'Expected Arrival Delay'
                result_value = f'{predicted_delay:.1f} minutes late'
                result_icon = '⏱️'
                card_class = 'result-delayed' if predicted_delay >= 15 else 'result-info'
                operational_status = 'Delayed' if predicted_delay >= 15 else 'Minor delay'
            else:
                result_title = 'Expected Early Arrival'
                result_value = f'{-predicted_delay:.1f} minutes early'
                result_icon = '✅'
                card_class = 'result-ontime'
                operational_status = 'On time'

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="result-kicker">Prediction result</div>
                    <div class="result-main">{result_icon} {escape(result_title)}</div>
                    <div class="result-grid">
                        <div class="result-stat">
                            <div class="result-stat-label">Estimated outcome</div>
                            <div class="result-stat-value">{escape(result_value)}</div>
                        </div>
                        <div class="result-stat">
                            <div class="result-stat-label">Flight status</div>
                            <div class="result-stat-value">{escape(operational_status)}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

