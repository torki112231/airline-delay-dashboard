import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# PAGE CONFIGURATION

st.set_page_config(
    page_title='Airline Delay Dashboard',
    page_icon='✈️',
    layout='wide'
)

sns.set_style('whitegrid')


# LOAD DATA AND MODELS

@st.cache_data
def load_data():
    st.write('Months in loaded file:')
    st.write(df['MONTH'].value_counts().sort_index())
    return pd.read_csv('small_clean_data.csv')


@st.cache_resource
def load_models():
    classification_model = joblib.load('xgb_classification_model.pkl')
    regression_model = joblib.load('xgb_regression_model.pkl')
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


# DASHBOARD TITLE

st.title('✈️ AIRLINE DELAY DASHBOARD')

st.write(
    'This dashboard analyzes airline flight delays, presents the main '
    'EDA findings, and uses XGBoost models for classification and regression.'
)


# SIDEBAR FILTERS

st.sidebar.header('FILTERS')

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
    '📊 Summary',
    '📈 EDA Charts',
    '🤖 Prediction'
])


# SUMMARY TAB

with tab1:

    st.subheader('SUMMARY')

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        'TOTAL FLIGHTS',
        f'{len(filtered_df):,}'
    )

    col2.metric(
        'AIRLINES',
        filtered_df[airline_filter_column].nunique()
    )

    col3.metric(
        'AVERAGE ARRIVAL DELAY',
        f'{filtered_df["ARRIVAL_DELAY"].mean():.2f} min'
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

    st.subheader('DATA PREVIEW')

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

    st.subheader('EDA CHARTS')

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
            df.groupby('MONTH')['ARRIVAL_DELAY']
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

    st.subheader('FLIGHT DELAY PREDICTION')

    prediction_type = st.radio(
        'Select Prediction Type',
        [
            'Classification',
            'Regression'
        ],
        horizontal=True
    )

    st.write('Enter the flight information below:')

    col1, col2 = st.columns(2)

    with col1:
        month = st.number_input(
            'Month',
            min_value=1,
            max_value=12,
            value=1
        )

        day = st.number_input(
            'Day',
            min_value=1,
            max_value=31,
            value=1
        )

        day_of_week = st.number_input(
            'Day of Week',
            min_value=1,
            max_value=7,
            value=1
        )

    with col2:
        scheduled_departure = st.number_input(
            'Scheduled Departure',
            min_value=0,
            max_value=2359,
            value=800
        )

        scheduled_time = st.number_input(
            'Scheduled Time (Minutes)',
            min_value=1,
            max_value=1000,
            value=120
        )

        distance = st.number_input(
            'Distance',
            min_value=1,
            max_value=6000,
            value=500
        )

    if prediction_type == 'Classification':

        if st.button('Predict Flight Status'):

            classification_input = pd.DataFrame(
                [[
                    month,
                    day,
                    day_of_week,
                    scheduled_departure,
                    scheduled_time,
                    distance
                ]],
                columns=[
                    'MONTH',
                    'DAY',
                    'DAY_OF_WEEK',
                    'SCHEDULED_DEPARTURE',
                    'SCHEDULED_TIME',
                    'DISTANCE'
                ]
            )

            prediction = model_classification.predict(
                classification_input
            )[0]

            if prediction == 1:
                st.error('Prediction: Delayed')
            else:
                st.success('Prediction: On Time')


    else:

        departure_delay = st.number_input(
            'Departure Delay (Minutes)',
            min_value=-100,
            max_value=2000,
            value=0
        )

        if st.button('Predict Arrival Delay'):

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

            st.info(
                f'Predicted Arrival Delay: {predicted_delay:.2f} minutes'
            )
