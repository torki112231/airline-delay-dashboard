
import base64
from datetime import date, time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Airline Delay Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


hero_b64 = image_b64(ASSETS / "hero_plane.png")
sidebar_b64 = image_b64(ASSETS / "sidebar_plane.png")

st.markdown(
    f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, #06101f 0%, #08172b 52%, #071321 100%);
            color: #eef4ff;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #071326 0%, #081a31 100%);
            border-right: 1px solid rgba(148, 163, 184, .16);
        }}

        [data-testid="stHeader"] {{
            background: rgba(6, 16, 31, .72);
            backdrop-filter: blur(10px);
        }}

        .block-container {{
            max-width: 1450px;
            padding-top: 1.15rem;
            padding-bottom: 2.2rem;
        }}

        .hero {{
            min-height: 205px;
            border-radius: 22px;
            padding: 2rem 2.25rem;
            background:
                linear-gradient(
                    90deg,
                    rgba(7, 25, 55, .99) 0%,
                    rgba(13, 35, 72, .95) 43%,
                    rgba(17, 28, 69, .30) 100%
                ),
                url("data:image/png;base64,{hero_b64}") center right / cover no-repeat;
            border: 1px solid rgba(100, 116, 139, .34);
            box-shadow: 0 18px 50px rgba(0, 0, 0, .24);
            margin-bottom: 1.4rem;
            overflow: hidden;
        }}

        .hero-kicker {{
            color: #9bdcff;
            font-size: .82rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }}

        .hero h1 {{
            color: white;
            font-size: clamp(2.2rem, 4vw, 3.35rem);
            line-height: 1.05;
            margin: .65rem 0 .7rem;
        }}

        .hero p {{
            max-width: 650px;
            color: #d1d9e8;
            font-size: 1rem;
            line-height: 1.65;
            margin: 0;
        }}

        .metric-card {{
            min-height: 132px;
            padding: 1.2rem;
            border-radius: 18px;
            background: linear-gradient(145deg, rgba(19, 35, 61, .97), rgba(12, 26, 48, .97));
            border: 1px solid rgba(100, 116, 139, .34);
            box-shadow: 0 14px 32px rgba(0, 0, 0, .18);
        }}

        .metric-label {{
            color: #aab7ca;
            font-size: .76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .07em;
        }}

        .metric-value {{
            color: white;
            font-size: 2rem;
            font-weight: 850;
            margin-top: .45rem;
        }}

        .metric-note {{
            color: #8bd7ff;
            font-size: .78rem;
            margin-top: .3rem;
        }}

        .section-title {{
            color: white;
            font-size: 1.2rem;
            font-weight: 800;
            margin: .35rem 0 .8rem;
        }}

        .chart-shell {{
            border-radius: 18px;
            padding: .45rem .8rem .8rem;
            background: rgba(10, 25, 47, .88);
            border: 1px solid rgba(100, 116, 139, .30);
        }}

        .prediction-box {{
            border-radius: 18px;
            padding: 1rem;
            background: rgba(8, 23, 43, .78);
            border: 1px solid rgba(100, 116, 139, .30);
        }}

        .result-good, .result-bad, .result-neutral {{
            margin-top: 1rem;
            border-radius: 15px;
            padding: 1rem 1.15rem;
            font-size: 1rem;
            font-weight: 700;
            border: 1px solid;
        }}

        .result-good {{
            background: rgba(16, 185, 129, .14);
            color: #69f0ae;
            border-color: rgba(52, 211, 153, .35);
        }}

        .result-bad {{
            background: rgba(239, 68, 68, .14);
            color: #ff8a8a;
            border-color: rgba(248, 113, 113, .36);
        }}

        .result-neutral {{
            background: rgba(14, 165, 233, .14);
            color: #8bd7ff;
            border-color: rgba(56, 189, 248, .34);
        }}

        .side-plane {{
            height: 135px;
            border-radius: 16px;
            margin-top: 1.2rem;
            background: url("data:image/png;base64,{sidebar_b64}") center / cover no-repeat;
            border: 1px solid rgba(99, 102, 241, .35);
        }}

        .side-copy {{
            padding: .9rem 0;
            color: #aab7ca;
            font-size: .86rem;
            line-height: 1.55;
        }}

        div[data-baseweb="tab-list"] {{
            gap: .4rem;
            background: rgba(10, 25, 47, .76);
            padding: .35rem;
            border-radius: 14px;
        }}

        button[data-baseweb="tab"] {{
            border-radius: 10px;
            padding: .5rem .9rem;
        }}

        div.stButton > button, div.stFormSubmitButton > button {{
            width: 100%;
            min-height: 2.8rem;
            border: 0;
            border-radius: 12px;
            color: white;
            font-weight: 800;
            background: linear-gradient(90deg, #5b4ce6, #7c3aed);
        }}

        [data-testid="stDataFrame"] {{
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(100, 116, 139, .28);
        }}

        #MainMenu, footer {{
            visibility: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    candidates = [
        ROOT / "data" / "small_clean_data_v2.csv",
        ROOT / "small_clean_data_v2.csv",
    ]

    for path in candidates:
        if path.exists():
            data = pd.read_csv(path)

            if "IS_DELAYED" not in data.columns and "ARRIVAL_DELAY" in data.columns:
                data["IS_DELAYED"] = (data["ARRIVAL_DELAY"] >= 15).astype(int)

            return data

    raise FileNotFoundError(
        "small_clean_data_v2.csv was not found inside data/ or the project root."
    )


@st.cache_resource(show_spinner=False)
def load_models():
    def load_first(paths):
        for path in paths:
            if path.exists():
                return joblib.load(path)
        return None

    classification = load_first(
        [
            ROOT / "models" / "xgb_classification_model.pkl",
            ROOT / "xgb_classification_model.pkl",
        ]
    )

    regression = load_first(
        [
            ROOT / "models" / "xgb_regression_model.pkl",
            ROOT / "xgb_regression_model.pkl",
        ]
    )

    return classification, regression


try:
    df = load_data()
except Exception as error:
    st.error(f"Data loading error: {error}")
    st.stop()

classification_model, regression_model = load_models()

AIRLINE = "AIRLINE_NAME" if "AIRLINE_NAME" in df.columns else "AIRLINE"
AIRPORT = (
    "ORIGIN_AIRPORT_NAME"
    if "ORIGIN_AIRPORT_NAME" in df.columns
    else "ORIGIN_AIRPORT"
)

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig, height=430):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=58, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,16,34,.15)",
        font=dict(color="#dbe7f7"),
        legend_title_text="",
    )
    return fig


def get_model_features(model):
    try:
        return model.get_booster().feature_names or []
    except Exception:
        return []


def build_classification_input(
    model,
    flight_date,
    airline,
    departure,
    scheduled_time,
    distance,
):
    features = get_model_features(model)

    if not features:
        raise ValueError("Classification model feature names are unavailable.")

    model_input = pd.DataFrame(0, index=[0], columns=features)

    values = {
        "MONTH": flight_date.month,
        "DAY": flight_date.day,
        "DAY_OF_WEEK": flight_date.isoweekday(),
        "SCHEDULED_DEPARTURE": departure.hour * 100 + departure.minute,
        "SCHEDULED_TIME": scheduled_time,
        "DISTANCE": distance,
    }

    for column, value in values.items():
        if column in model_input.columns:
            model_input.loc[0, column] = value

    for dummy in [f"AIRLINE_NAME_{airline}", f"AIRLINE_{airline}"]:
        if dummy in model_input.columns:
            model_input.loc[0, dummy] = 1
            break

    return model_input


# Realistic ranges based on the middle 98% of the actual dataset.
scheduled_time_min = max(20, int(df["SCHEDULED_TIME"].quantile(0.01)))
scheduled_time_max = min(720, int(df["SCHEDULED_TIME"].quantile(0.99)))
distance_min = max(30, int(df["DISTANCE"].quantile(0.01)))
distance_max = min(5000, int(df["DISTANCE"].quantile(0.99)))
departure_delay_min = max(-60, int(df["DEPARTURE_DELAY"].quantile(0.01)))
departure_delay_max = min(600, int(df["DEPARTURE_DELAY"].quantile(0.99)))

with st.sidebar:
    st.markdown("## ✈️ Airline Delay")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Prediction", "Dataset"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    selected_month = st.selectbox(
        "Select Month",
        ["All Months"] + list(range(1, 13)),
        format_func=lambda value: (
            value if value == "All Months" else MONTH_NAMES[int(value)]
        ),
    )

    selected_airline = st.selectbox(
        "Select Airline",
        ["All Airlines"]
        + sorted(df[AIRLINE].dropna().astype(str).unique().tolist()),
    )

    st.markdown('<div class="side-plane"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="side-copy">
            <b>Turning Data Into Smarter Flights</b><br>
            Explore delay patterns and compare airline performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

filtered_df = df.copy()

if selected_month != "All Months":
    filtered_df = filtered_df[filtered_df["MONTH"] == selected_month]

if selected_airline != "All Airlines":
    filtered_df = filtered_df[
        filtered_df[AIRLINE].astype(str) == selected_airline
    ]

if filtered_df.empty:
    st.warning("No flights match the selected filters.")
    st.stop()


if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Welcome to</div>
            <h1>Airline Delay Dashboard</h1>
            <p>
                Analyze flight performance, compare airlines and explore
                delay patterns using real flight data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    delayed_rate = filtered_df["IS_DELAYED"].mean() * 100

    card_columns = st.columns(4)
    card_values = [
        (
            "Total Flights",
            f"{len(filtered_df):,}",
            "Flights in the current selection",
        ),
        (
            "Average Delay",
            f"{filtered_df['ARRIVAL_DELAY'].mean():.1f} min",
            "Average arrival delay",
        ),
        (
            "On-Time Rate",
            f"{100 - delayed_rate:.1f}%",
            "Arrival delay below 15 minutes",
        ),
        (
            "Delayed Flights",
            f"{delayed_rate:.1f}%",
            "Arrival delay of 15 minutes or more",
        ),
    ]

    for column, item in zip(card_columns, card_values):
        with column:
            metric_card(*item)

    st.markdown("")

    monthly = (
        filtered_df.groupby("MONTH", as_index=False)["ARRIVAL_DELAY"]
        .mean()
        .sort_values("MONTH")
    )
    monthly["Month"] = monthly["MONTH"].map(
        lambda value: MONTH_NAMES[int(value)][:3]
    )

    delayed_airlines = (
        filtered_df.groupby(AIRLINE, as_index=False)["ARRIVAL_DELAY"]
        .mean()
        .nlargest(10, "ARRIVAL_DELAY")
        .sort_values("ARRIVAL_DELAY")
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        fig = px.line(
            monthly,
            x="Month",
            y="ARRIVAL_DELAY",
            markers=True,
            title="Average Delay Trend by Month",
            labels={"ARRIVAL_DELAY": "Average Delay (minutes)"},
        )
        fig.update_traces(line_width=4, marker_size=9)
        st.plotly_chart(style_figure(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_right:
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        fig = px.bar(
            delayed_airlines,
            x="ARRIVAL_DELAY",
            y=AIRLINE,
            orientation="h",
            title="Top 10 Most Delayed Airlines",
            labels={"ARRIVAL_DELAY": "Average Delay (minutes)"},
        )
        st.plotly_chart(style_figure(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Prediction":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Machine Learning</div>
            <h1>Predict Flight Delay</h1>
            <p>
                Use the classification model to estimate delay risk or the
                regression model to estimate the expected arrival delay.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    classification_tab, regression_tab = st.tabs(
        ["Delay Risk", "Expected Delay Minutes"]
    )

    with classification_tab:
        if classification_model is None:
            st.error("Classification model was not found.")
        else:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

            with st.form("classification_form"):
                column_1, column_2, column_3 = st.columns(3)

                with column_1:
                    flight_date = st.date_input(
                        "Flight Date",
                        value=date(2015, 7, 15),
                        key="classification_date",
                    )
                    airline = st.selectbox(
                        "Airline",
                        sorted(df[AIRLINE].dropna().astype(str).unique()),
                        key="classification_airline",
                    )

                with column_2:
                    departure = st.time_input(
                        "Scheduled Departure",
                        value=time(8, 0),
                        key="classification_departure",
                    )
                    scheduled_time = st.slider(
                        "Scheduled Time (minutes)",
                        min_value=scheduled_time_min,
                        max_value=scheduled_time_max,
                        value=int(df["SCHEDULED_TIME"].median()),
                        step=5,
                    )

                with column_3:
                    distance = st.slider(
                        "Distance (miles)",
                        min_value=distance_min,
                        max_value=distance_max,
                        value=int(df["DISTANCE"].median()),
                        step=10,
                        key="classification_distance",
                    )

                classify = st.form_submit_button("Predict Delay Risk")

            if classify:
                try:
                    model_input = build_classification_input(
                        classification_model,
                        flight_date,
                        airline,
                        departure,
                        scheduled_time,
                        distance,
                    )

                    prediction = int(
                        classification_model.predict(model_input)[0]
                    )

                    probability = (
                        float(
                            classification_model.predict_proba(model_input)[
                                0, 1
                            ]
                        )
                        * 100
                    )

                    if prediction == 1:
                        st.markdown(
                            f"""
                            <div class="result-bad">
                                Likely Delayed — Chance of delay: {probability:.1f}%
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-good">
                                Likely On Time — Chance of delay: {probability:.1f}%
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as error:
                    st.error(f"Prediction error: {error}")

            st.markdown("</div>", unsafe_allow_html=True)

    with regression_tab:
        if regression_model is None:
            st.error("Regression model was not found.")
        else:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

            with st.form("regression_form"):
                column_1, column_2, column_3 = st.columns(3)

                with column_1:
                    regression_date = st.date_input(
                        "Flight Date",
                        value=date(2015, 7, 15),
                        key="regression_date",
                    )

                with column_2:
                    departure_delay = st.slider(
                        "Departure Delay (minutes)",
                        min_value=departure_delay_min,
                        max_value=departure_delay_max,
                        value=0,
                        step=5,
                    )

                with column_3:
                    regression_distance = st.slider(
                        "Distance (miles)",
                        min_value=distance_min,
                        max_value=distance_max,
                        value=int(df["DISTANCE"].median()),
                        step=10,
                        key="regression_distance",
                    )

                regress = st.form_submit_button(
                    "Predict Arrival Delay"
                )

            if regress:
                regression_input = pd.DataFrame(
                    [
                        [
                            departure_delay,
                            regression_distance,
                            regression_date.month,
                            regression_date.day,
                            regression_date.isoweekday(),
                        ]
                    ],
                    columns=[
                        "DEPARTURE_DELAY",
                        "DISTANCE",
                        "MONTH",
                        "DAY",
                        "DAY_OF_WEEK",
                    ],
                )

                try:
                    predicted_delay = float(
                        regression_model.predict(regression_input)[0]
                    )

                    if predicted_delay >= 15:
                        st.markdown(
                            f"""
                            <div class="result-bad">
                                Expected Arrival Delay: {predicted_delay:.1f} minutes late
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif predicted_delay > 0:
                        st.markdown(
                            f"""
                            <div class="result-neutral">
                                Minor Expected Delay: {predicted_delay:.1f} minutes late
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-good">
                                Expected Early Arrival: {-predicted_delay:.1f} minutes early
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as error:
                    st.error(f"Prediction error: {error}")

            st.markdown("</div>", unsafe_allow_html=True)


else:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Data Explorer</div>
            <h1>Flight Dataset</h1>
            <p>
                View the cleaned flight records for the selected month
                and airline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-title">{len(filtered_df):,} flight records</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=620,
        hide_index=True,
    )

st.caption("© 2025 Airline Delay Dashboard")
