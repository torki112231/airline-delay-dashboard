
import base64
from pathlib import Path
from datetime import date, time

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Airline Delay Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"

def image_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

hero_b64 = image_b64(ASSETS / "hero_plane.png")
side_b64 = image_b64(ASSETS / "sidebar_plane.png")

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(135deg,#06101f 0%,#08162a 48%,#07111f 100%);
    color:#eef4ff;
}}
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,#071326 0%,#081a31 100%);
    border-right:1px solid rgba(148,163,184,.16);
}}
[data-testid="stHeader"] {{background:rgba(6,16,31,.72);backdrop-filter:blur(10px)}}
.block-container {{max-width:1450px;padding-top:1.15rem;padding-bottom:2rem}}

.hero {{
    min-height:195px;
    border-radius:22px;
    padding:2rem 2.2rem;
    background:
      linear-gradient(90deg,rgba(8,25,55,.96) 0%,rgba(15,37,74,.86) 42%,rgba(20,26,72,.35) 100%),
      url("data:image/png;base64,{hero_b64}") center right/cover no-repeat;
    border:1px solid rgba(100,116,139,.35);
    box-shadow:0 18px 50px rgba(0,0,0,.25);
    margin-bottom:1.4rem;
}}
.hero-small {{color:#9bdcff;font-size:.82rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}}
.hero h1 {{font-size:3rem;line-height:1.05;margin:.65rem 0 .65rem;color:white}}
.hero p {{max-width:650px;color:#d1d9e8;font-size:1rem;line-height:1.7}}

.metric {{
    min-height:130px;padding:1.2rem;border-radius:18px;
    background:linear-gradient(145deg,rgba(19,35,61,.96),rgba(12,26,48,.96));
    border:1px solid rgba(100,116,139,.35);
    box-shadow:0 14px 32px rgba(0,0,0,.18);
}}
.metric-label {{color:#aab7ca;font-size:.78rem;font-weight:750;text-transform:uppercase;letter-spacing:.07em}}
.metric-value {{color:white;font-size:2rem;font-weight:850;margin-top:.45rem}}
.metric-note {{color:#8bd7ff;font-size:.78rem;margin-top:.3rem}}
.chart-card {{
    border-radius:18px;padding:.65rem 1rem 1rem;
    background:rgba(10,25,47,.86);
    border:1px solid rgba(100,116,139,.32);
}}
.insight {{
    border-radius:13px;padding:.85rem 1rem;margin-top:.5rem;
    background:rgba(33,44,78,.62);border:1px solid rgba(99,102,241,.28);
    color:#dce7f7;font-size:.9rem;
}}
.bottom-stat {{
    padding:1.1rem;border-radius:16px;background:rgba(11,28,52,.9);
    border:1px solid rgba(100,116,139,.3)
}}
.bottom-stat b {{font-size:1.45rem;color:white}}
.side-image {{
    height:125px;border-radius:16px;margin-top:1.2rem;
    background:url("data:image/png;base64,{side_b64}") center/cover no-repeat;
    border:1px solid rgba(99,102,241,.35);
}}
.side-copy {{padding:.9rem 0;color:#aab7ca;font-size:.86rem;line-height:1.55}}
div[data-baseweb="radio"] > div {{gap:.35rem}}
div.stButton>button {{
    width:100%;min-height:3rem;border:0;border-radius:12px;color:white;font-weight:800;
    background:linear-gradient(90deg,#5b4ce6,#7c3aed);
}}
[data-testid="stDataFrame"] {{border-radius:15px;overflow:hidden}}
#MainMenu, footer {{visibility:hidden}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    for p in [ROOT/"data"/"small_clean_data_v2.csv", ROOT/"small_clean_data_v2.csv"]:
        if p.exists():
            d = pd.read_csv(p)
            if "IS_DELAYED" not in d.columns:
                d["IS_DELAYED"] = (d["ARRIVAL_DELAY"] >= 15).astype(int)
            return d
    raise FileNotFoundError("small_clean_data_v2.csv not found.")

@st.cache_resource(show_spinner=False)
def load_models():
    def load_first(paths):
        for p in paths:
            if p.exists():
                return joblib.load(p)
        return None
    return (
        load_first([ROOT/"models"/"xgb_classification_model.pkl", ROOT/"xgb_classification_model.pkl"]),
        load_first([ROOT/"models"/"xgb_regression_model.pkl", ROOT/"xgb_regression_model.pkl"])
    )

df = load_data()
clf, reg = load_models()
AIRLINE = "AIRLINE_NAME" if "AIRLINE_NAME" in df.columns else "AIRLINE"
AIRPORT = "ORIGIN_AIRPORT_NAME" if "ORIGIN_AIRPORT_NAME" in df.columns else "ORIGIN_AIRPORT"

def metric(label, value, note):
    st.markdown(f"""<div class="metric">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-note">{note}</div></div>""", unsafe_allow_html=True)

def style(fig, h=420):
    fig.update_layout(
        template="plotly_dark", height=h,
        margin=dict(l=20,r=20,t=55,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,16,34,.15)",
        font=dict(color="#dbe7f7"),
        showlegend=False
    )
    return fig

with st.sidebar:
    st.markdown("## ✈️ Airline Delay")
    st.markdown("### Dashboard")
    page = st.radio("Navigation", ["Dashboard","Prediction","Dataset"], label_visibility="collapsed")
    st.markdown("---")
    month = st.selectbox("Select Month", ["All Months"] + list(range(1,13)),
                         format_func=lambda x: x if x=="All Months" else pd.Timestamp(2015,int(x),1).strftime("%B"))
    airline = st.selectbox("Select Airline", ["All Airlines"] + sorted(df[AIRLINE].dropna().astype(str).unique().tolist()))
    st.markdown(f'<div class="side-image"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy"><b>Turning Data Into Smarter Flights</b><br>Insights that help airlines understand delay patterns and improve performance.</div>', unsafe_allow_html=True)

filtered = df.copy()
if month != "All Months":
    filtered = filtered[filtered["MONTH"] == month]
if airline != "All Airlines":
    filtered = filtered[filtered[AIRLINE].astype(str) == airline]

if page == "Dashboard":
    st.markdown("""<div class="hero">
    <div class="hero-small">Welcome to</div>
    <h1>Airline Delay Dashboard</h1>
    <p>Explore insights, analyze trends and understand flight delays using real data and machine learning.</p>
    </div>""", unsafe_allow_html=True)

    delayed_rate = filtered["IS_DELAYED"].mean()*100
    cols = st.columns(4)
    cards = [
        ("Total Flights", f"{len(filtered):,}", "Total number of flights"),
        ("Average Delay", f"{filtered['ARRIVAL_DELAY'].mean():.1f} min", "Average arrival delay"),
        ("On-Time Rate", f"{100-delayed_rate:.1f}%", "Flights arriving on time"),
        ("Delayed Flights", f"{delayed_rate:.1f}%", "Flights delayed ≥ 15 min"),
    ]
    for c, item in zip(cols, cards):
        with c: metric(*item)

    monthly = filtered.groupby("MONTH",as_index=False)["ARRIVAL_DELAY"].mean()
    monthly["Month"] = monthly["MONTH"].apply(lambda x: pd.Timestamp(2015,int(x),1).strftime("%b"))
    top_airlines = (filtered.groupby(AIRLINE,as_index=False)["ARRIVAL_DELAY"]
                    .mean().nlargest(10,"ARRIVAL_DELAY").sort_values("ARRIVAL_DELAY"))

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = px.line(monthly,x="Month",y="ARRIVAL_DELAY",markers=True,
                      title="Average Delay Trend by Month",
                      labels={"ARRIVAL_DELAY":"Delay (minutes)"})
        fig.update_traces(line=dict(width=4),marker=dict(size=9))
        st.plotly_chart(style(fig),use_container_width=True)
        if not monthly.empty:
            peak = monthly.loc[monthly["ARRIVAL_DELAY"].idxmax()]
            st.markdown(f'<div class="insight">💡 Delays peak in <b>{peak["Month"]}</b> at an average of <b>{peak["ARRIVAL_DELAY"]:.1f} minutes</b>.</div>',unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = px.bar(top_airlines,x="ARRIVAL_DELAY",y=AIRLINE,orientation="h",
                     title="Top 10 Airlines by Average Delay",
                     labels={"ARRIVAL_DELAY":"Average Delay (minutes)"})
        st.plotly_chart(style(fig),use_container_width=True)
        if not top_airlines.empty:
            worst = top_airlines.iloc[-1]
            st.markdown(f'<div class="insight">💡 <b>{worst[AIRLINE]}</b> has the highest average delay in the current selection.</div>',unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    busy = filtered["ORIGIN_AIRPORT"].mode().iloc[0] if "ORIGIN_AIRPORT" in filtered and not filtered.empty else "-"
    delayed_airport = (filtered.groupby("ORIGIN_AIRPORT")["ARRIVAL_DELAY"].mean().idxmax()
                       if "ORIGIN_AIRPORT" in filtered and not filtered.empty else "-")
    avg_distance = filtered["DISTANCE"].mean()
    avg_time = filtered["AIR_TIME"].mean() if "AIR_TIME" in filtered else filtered["SCHEDULED_TIME"].mean()

    st.markdown("")
    b1,b2,b3,b4 = st.columns(4)
    for c,title,value,note in [
        (b1,"Busiest Airport",busy,"Most frequent origin"),
        (b2,"Most Delayed Airport",delayed_airport,"Highest average delay"),
        (b3,"Average Distance",f"{avg_distance:,.0f} mi","Average route distance"),
        (b4,"Average Flight Time",f"{avg_time:,.0f} min","Average flight duration"),
    ]:
        with c:
            st.markdown(f'<div class="bottom-stat">{title}<br><b>{value}</b><br><span style="color:#93a4ba;font-size:.8rem">{note}</span></div>',unsafe_allow_html=True)

elif page == "Prediction":
    st.markdown("""<div class="hero">
    <div class="hero-small">Machine Learning</div>
    <h1>Predict Flight Delay</h1>
    <p>Enter the flight details below to estimate whether the flight is likely to be delayed.</p>
    </div>""", unsafe_allow_html=True)

    if clf is None:
        st.error("Classification model not found.")
    else:
        with st.form("prediction"):
            c1,c2,c3 = st.columns(3)
            with c1:
                d = st.date_input("Flight Date", date(2015,7,15))
                chosen_airline = st.selectbox("Airline", sorted(df[AIRLINE].dropna().astype(str).unique()))
            with c2:
                departure = st.time_input("Scheduled Departure", time(8,0))
                duration = st.slider("Scheduled Time (minutes)",30,600,120,5)
            with c3:
                distance = st.slider("Distance (miles)",int(df["DISTANCE"].min()),int(df["DISTANCE"].max()),int(df["DISTANCE"].median()),10)
            submit = st.form_submit_button("Predict Delay")

        if submit:
            try:
                features = clf.get_booster().feature_names
                x = pd.DataFrame(0,index=[0],columns=features)
                values = {
                    "MONTH":d.month,"DAY":d.day,"DAY_OF_WEEK":d.isoweekday(),
                    "SCHEDULED_DEPARTURE":departure.hour*100+departure.minute,
                    "SCHEDULED_TIME":duration,"DISTANCE":distance
                }
                for k,v in values.items():
                    if k in x.columns: x.loc[0,k]=v
                for name in [f"AIRLINE_NAME_{chosen_airline}",f"AIRLINE_{chosen_airline}"]:
                    if name in x.columns:
                        x.loc[0,name]=1
                        break
                pred = int(clf.predict(x)[0])
                prob = float(clf.predict_proba(x)[0,1])*100
                if pred:
                    st.error(f"Likely Delayed — Chance of delay: {prob:.1f}%")
                else:
                    st.success(f"Likely On Time — Chance of delay: {prob:.1f}%")
            except Exception as e:
                st.error(f"Prediction error: {e}")

else:
    st.markdown("""<div class="hero">
    <div class="hero-small">Data Explorer</div>
    <h1>Flight Dataset</h1>
    <p>Preview the cleaned dataset used for analysis and machine-learning model training.</p>
    </div>""", unsafe_allow_html=True)
    st.metric("Rows in current selection", f"{len(filtered):,}")
    default_cols = [c for c in [AIRLINE,AIRPORT,"DESTINATION_AIRPORT","MONTH","DAY","DISTANCE","DEPARTURE_DELAY","ARRIVAL_DELAY","IS_DELAYED"] if c in filtered.columns]
    cols = st.multiselect("Columns", filtered.columns.tolist(), default=default_cols)
    if cols:
        st.dataframe(filtered[cols],use_container_width=True,height=560,hide_index=True)
        st.download_button("Download filtered data",filtered[cols].to_csv(index=False).encode(),"filtered_flights.csv","text/csv")

st.caption("© 2025 Airline Delay Dashboard")
