import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import date


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

    st.subheader('FLIGHT DELAY PREDICTION')

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
            scheduled_time = st.number_input(
                'Scheduled Flight Duration (minutes)',
                min_value=1,
                max_value=1000,
                value=120,
                help='Enter the planned flight duration in minutes.'
            )

            distance = st.number_input(
                'Flight Distance (miles)',
                min_value=1,
                max_value=6000,
                value=500,
                help='Enter the flight distance in miles.'
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

            st.markdown('### Prediction Result')

            if prediction == 1:
                st.error(
                    f'Likely Delayed — estimated delay probability: '
                    f'{delay_probability:.1f}%'
                )
            else:
                st.success(
                    f'Likely On Time — estimated delay probability: '
                    f'{delay_probability:.1f}%'
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

            distance = st.number_input(
                'Flight Distance (miles)',
                min_value=1,
                max_value=6000,
                value=500,
                key='regression_distance',
                help='Enter the flight distance in miles.'
            )

        with col2:
            departure_delay = st.number_input(
                'Departure Delay (minutes)',
                min_value=-100,
                max_value=2000,
                value=0,
                help=(
                    'Use 0 when the flight departed on time. '
                    'Negative values mean the flight departed early.'
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

            st.markdown('### Prediction Result')

            if predicted_delay > 0:
                st.info(
                    f'Expected arrival delay: {predicted_delay:.1f} minutes'
                )
            else:
                st.success(
                    f'Expected to arrive {-predicted_delay:.1f} minutes early'
                )

