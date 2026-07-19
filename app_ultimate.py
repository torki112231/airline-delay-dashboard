import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, time
from pathlib import Path

st.set_page_config(page_title="Airline Delay Intelligence", page_icon="✈️", layout="wide")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 10% 5%,rgba(14,165,233,.12),transparent 24rem),radial-gradient(circle at 90% 15%,rgba(99,102,241,.12),transparent 24rem),#07111f;color:#e5eef9}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071827,#0b2033);border-right:1px solid rgba(148,163,184,.16)}
[data-testid="stHeader"]{background:rgba(7,17,31,.75);backdrop-filter:blur(12px)}
.block-container{max-width:1500px;padding-top:1.4rem;padding-bottom:3rem}
.hero{position:relative;overflow:hidden;padding:2rem 2.2rem;border-radius:26px;background:linear-gradient(120deg,rgba(14,165,233,.22),rgba(79,70,229,.22)),rgba(15,23,42,.9);border:1px solid rgba(125,211,252,.22);box-shadow:0 24px 70px rgba(0,0,0,.28);margin-bottom:1.3rem}
.hero:after{content:"✈";position:absolute;right:3rem;top:0;font-size:8rem;opacity:.08;transform:rotate(-12deg)}
.eyebrow{color:#7dd3fc;font-size:.82rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase}
.hero h1{color:white;margin:.25rem 0 .35rem;font-size:clamp(2rem,4vw,3.6rem);line-height:1.05}
.hero p{color:#cbd5e1;max-width:850px;margin:0;font-size:1.02rem}
.section-title{font-size:1.35rem;font-weight:800;color:#f8fafc;margin:.35rem 0 1rem}
.metric-card{min-height:125px;padding:1.15rem 1.2rem;border-radius:20px;background:linear-gradient(145deg,rgba(30,41,59,.96),rgba(15,23,42,.96));border:1px solid rgba(148,163,184,.16);box-shadow:0 16px 35px rgba(0,0,0,.2)}
.metric-label{color:#94a3b8;font-size:.76rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.metric-value{color:white;font-size:1.8rem;font-weight:850;margin-top:.35rem}
.metric-note{color:#7dd3fc;font-size:.78rem;margin-top:.25rem}
.insight{padding:1.1rem 1.2rem;border-radius:18px;background:rgba(15,23,42,.78);border:1px solid rgba(148,163,184,.14);margin-bottom:.75rem}
.insight strong{color:#7dd3fc}
.good,.bad,.neutral{padding:1.5rem;border-radius:22px;text-align:center;border:1px solid;margin-top:1rem}
.good{background:rgba(16,185,129,.12);border-color:rgba(52,211,153,.4)}
.bad{background:rgba(239,68,68,.12);border-color:rgba(248,113,113,.42)}
.neutral{background:rgba(14,165,233,.12);border-color:rgba(56,189,248,.4)}
.result-title{font-size:2rem;font-weight:900;color:white}.result-subtitle{color:#cbd5e1;margin-top:.35rem}
div[data-baseweb="tab-list"]{gap:.55rem;background:rgba(15,23,42,.65);padding:.45rem;border-radius:16px;border:1px solid rgba(148,163,184,.13)}
button[data-baseweb="tab"]{border-radius:12px;padding:.55rem 1rem}
div.stButton>button,div.stDownloadButton>button{width:100%;min-height:3rem;border:0;border-radius:14px;font-weight:850;color:white;background:linear-gradient(90deg,#0284c7,#4f46e5);box-shadow:0 12px 25px rgba(37,99,235,.22)}
[data-testid="stDataFrame"]{border-radius:16px;overflow:hidden;border:1px solid rgba(148,163,184,.14)}
#MainMenu,footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

MONTHS={1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
SHORT={k:v[:3] for k,v in MONTHS.items()}

@st.cache_data(show_spinner=False)
def load_data():
    for p in [Path("data/small_clean_data_v2.csv"),Path("small_clean_data_v2.csv")]:
        if p.exists():
            d=pd.read_csv(p)
            if "IS_DELAYED" not in d and "ARRIVAL_DELAY" in d:
                d["IS_DELAYED"]=(d["ARRIVAL_DELAY"]>=15).astype(int)
            return d
    raise FileNotFoundError("small_clean_data_v2.csv was not found.")

@st.cache_resource(show_spinner=False)
def load_models():
    def first(paths):
        for p in paths:
            if p.exists(): return joblib.load(p)
        return None
    return (
        first([Path("models/xgb_classification_model.pkl"),Path("xgb_classification_model.pkl")]),
        first([Path("models/xgb_regression_model.pkl"),Path("xgb_regression_model.pkl")])
    )

try:
    df=load_data()
except Exception as e:
    st.error(f"Data loading error: {e}"); st.stop()

clf,reg=load_models()
AIRLINE="AIRLINE_NAME" if "AIRLINE_NAME" in df else "AIRLINE"
AIRPORT="ORIGIN_AIRPORT_NAME" if "ORIGIN_AIRPORT_NAME" in df else "ORIGIN_AIRPORT"

def mean(d,c):
    return 0.0 if d.empty or c not in d else float(d[c].mean())

def card(label,value,note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

def polish(fig,h=430):
    fig.update_layout(template="plotly_dark",height=h,margin=dict(l=20,r=20,t=65,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(15,23,42,.45)",font=dict(color="#dbeafe"),legend_title_text="")
    return fig

def model_features(model):
    try:return model.get_booster().feature_names or []
    except:return []

def classification_row(model,month,day,dow,airline,departure,duration,distance):
    cols=model_features(model)
    if not cols: raise ValueError("Model feature names are unavailable.")
    x=pd.DataFrame(0,index=[0],columns=cols)
    for k,v in {"MONTH":month,"DAY":day,"DAY_OF_WEEK":dow,"SCHEDULED_DEPARTURE":departure,"SCHEDULED_TIME":duration,"DISTANCE":distance}.items():
        if k in x:x.loc[0,k]=v
    for c in [f"AIRLINE_NAME_{airline}",f"AIRLINE_{airline}"]:
        if c in x:x.loc[0,c]=1;break
    return x

with st.sidebar:
    st.markdown("## ✈️ Flight Intelligence")
    st.caption("Explore performance and predict delays.")
    st.markdown("---")
    months=st.multiselect("Month",sorted(df.MONTH.dropna().astype(int).unique()),format_func=lambda x:MONTHS[int(x)],placeholder="All months")
    airlines=st.multiselect("Airline",sorted(df[AIRLINE].dropna().astype(str).unique()),placeholder="All airlines")
    airports=st.multiselect("Origin Airport",sorted(df[AIRPORT].dropna().astype(str).unique()),placeholder="All airports")
    low,high=int(max(0,df.DISTANCE.min())),int(df.DISTANCE.max())
    distance_range=st.slider("Distance Range (miles)",low,high,(low,high))
    st.markdown("---")
    st.caption("All charts update instantly.")

f=df.copy()
if months:f=f[f.MONTH.isin(months)]
if airlines:f=f[f[AIRLINE].astype(str).isin(airlines)]
if airports:f=f[f[AIRPORT].astype(str).isin(airports)]
f=f[f.DISTANCE.between(*distance_range)]

st.markdown("""<div class="hero"><div class="eyebrow">US Flight Performance • 2015</div><h1>Airline Delay Intelligence</h1><p>Interactive analytics and machine learning for comparing airlines, investigating airport risk, and estimating flight delays.</p></div>""",unsafe_allow_html=True)

if f.empty:
    st.warning("No flights match the selected filters.");st.stop()

overview,airline_tab,airport_tab,prediction,data_tab=st.tabs(["🏠 Executive Overview","✈️ Airline Intelligence","🛫 Airport Analysis","🤖 Smart Prediction","🧾 Data Explorer"])

with overview:
    st.markdown('<div class="section-title">Performance Snapshot</div>',unsafe_allow_html=True)
    delayed=mean(f,"IS_DELAYED")*100
    cols=st.columns(5)
    values=[("Total Flights",f"{len(f):,}","Filtered records"),("Average Delay",f'{mean(f,"ARRIVAL_DELAY"):.1f} min',"Arrival performance"),("On-Time Rate",f"{100-delayed:.1f}%","Delay below 15 min"),("Delayed Flights",f"{delayed:.1f}%","Delay of 15+ min"),("Average Distance",f'{mean(f,"DISTANCE"):,.0f} mi',"Typical route length")]
    for c,v in zip(cols,values):
        with c:card(*v)
    monthly=f.groupby("MONTH",as_index=False).agg(Average_Delay=("ARRIVAL_DELAY","mean"),Flights=("ARRIVAL_DELAY","size"),Delay_Rate=("IS_DELAYED","mean")).sort_values("MONTH")
    monthly["Month"]=monthly.MONTH.map(SHORT);monthly["Delay_Rate"]*=100
    c1,c2=st.columns([1.5,1])
    with c1:
        fig=px.line(monthly,x="Month",y="Average_Delay",markers=True,title="Monthly Arrival Delay Trend",labels={"Average_Delay":"Average Delay (minutes)"})
        fig.update_traces(line_width=4,marker_size=9);st.plotly_chart(polish(fig),use_container_width=True)
    with c2:
        status=f.IS_DELAYED.map({0:"On Time",1:"Delayed"}).value_counts().rename_axis("Status").reset_index(name="Flights")
        fig=px.pie(status,names="Status",values="Flights",hole=.68,title="Flight Status Mix");fig.update_traces(textinfo="percent+label")
        st.plotly_chart(polish(fig),use_container_width=True)
    c1,c2=st.columns(2)
    airline_perf=f.groupby(AIRLINE,as_index=False).agg(Flights=("ARRIVAL_DELAY","size"),Average_Delay=("ARRIVAL_DELAY","mean"),Delay_Rate=("IS_DELAYED","mean"))
    airline_perf["Delay_Rate"]*=100
    with c1:
        fig=px.scatter(airline_perf,x="Flights",y="Average_Delay",size="Delay_Rate",hover_name=AIRLINE,title="Airline Volume vs Average Delay",labels={"Average_Delay":"Average Delay (minutes)","Delay_Rate":"Delayed Flights (%)"})
        st.plotly_chart(polish(fig),use_container_width=True)
    with c2:
        airports_top=f.groupby(AIRPORT,as_index=False).agg(Average_Delay=("ARRIVAL_DELAY","mean"),Flights=("ARRIVAL_DELAY","size")).query("Flights>=20").nlargest(10,"Average_Delay").sort_values("Average_Delay")
        fig=px.bar(airports_top,x="Average_Delay",y=AIRPORT,orientation="h",title="Highest-Delay Origin Airports",labels={"Average_Delay":"Average Delay (minutes)"})
        st.plotly_chart(polish(fig),use_container_width=True)

with airline_tab:
    st.markdown('<div class="section-title">Airline Performance Leaderboard</div>',unsafe_allow_html=True)
    stats=f.groupby(AIRLINE,as_index=False).agg(Flights=("ARRIVAL_DELAY","size"),Average_Arrival_Delay=("ARRIVAL_DELAY","mean"),Average_Departure_Delay=("DEPARTURE_DELAY","mean"),On_Time_Rate=("IS_DELAYED",lambda x:(1-x.mean())*100),Average_Distance=("DISTANCE","mean"))
    minimum=st.slider("Minimum flights required",1,max(1,int(stats.Flights.max())),min(100,max(1,int(stats.Flights.max()))))
    comp=stats[stats.Flights>=minimum]
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(comp.sort_values("Average_Arrival_Delay"),x="Average_Arrival_Delay",y=AIRLINE,orientation="h",title="Average Arrival Delay by Airline",labels={"Average_Arrival_Delay":"Average Delay (minutes)"})
        st.plotly_chart(polish(fig,500),use_container_width=True)
    with c2:
        fig=px.bar(comp.sort_values("On_Time_Rate"),x="On_Time_Rate",y=AIRLINE,orientation="h",title="On-Time Rate by Airline",labels={"On_Time_Rate":"On-Time Rate (%)"})
        st.plotly_chart(polish(fig,500),use_container_width=True)
    st.dataframe(comp.style.format({"Flights":"{:,.0f}","Average_Arrival_Delay":"{:.2f}","Average_Departure_Delay":"{:.2f}","On_Time_Rate":"{:.2f}%","Average_Distance":"{:,.0f}"}),use_container_width=True,hide_index=True)

with airport_tab:
    st.markdown('<div class="section-title">Origin Airport Risk Analysis</div>',unsafe_allow_html=True)
    a=f.groupby(AIRPORT,as_index=False).agg(Flights=("ARRIVAL_DELAY","size"),Average_Delay=("ARRIVAL_DELAY","mean"),Delay_Rate=("IS_DELAYED","mean"),Average_Taxi_Out=("TAXI_OUT","mean"))
    a.Delay_Rate*=100
    minimum=st.slider("Minimum flights per airport",1,max(1,int(a.Flights.max())),min(50,max(1,int(a.Flights.max()))),key="airport_min")
    ac=a[a.Flights>=minimum]
    c1,c2=st.columns([1.2,1])
    with c1:
        fig=px.scatter(ac,x="Flights",y="Average_Delay",size="Delay_Rate",hover_name=AIRPORT,title="Airport Traffic vs Delay",labels={"Average_Delay":"Average Delay (minutes)","Delay_Rate":"Delayed Flights (%)"})
        st.plotly_chart(polish(fig,500),use_container_width=True)
    with c2:
        top=ac.nlargest(12,"Delay_Rate").sort_values("Delay_Rate")
        fig=px.bar(top,x="Delay_Rate",y=AIRPORT,orientation="h",title="Highest Airport Delay Rates",labels={"Delay_Rate":"Delayed Flights (%)"})
        st.plotly_chart(polish(fig,500),use_container_width=True)
    choice=st.selectbox("Inspect one origin airport",sorted(f[AIRPORT].dropna().astype(str).unique()))
    d=f[f[AIRPORT].astype(str)==choice]
    cs=st.columns(4)
    vals=[("Flights",f"{len(d):,}",choice),("Average Delay",f'{mean(d,"ARRIVAL_DELAY"):.1f} min',"Arrival performance"),("Delay Rate",f'{mean(d,"IS_DELAYED")*100:.1f}%',"15+ minute delays"),("Taxi-Out",f'{mean(d,"TAXI_OUT"):.1f} min',"Average runway wait")]
    for c,v in zip(cs,vals):
        with c:card(*v)

with prediction:
    st.markdown('<div class="section-title">AI-Powered Flight Prediction</div>',unsafe_allow_html=True)
    class_tab,reg_tab=st.tabs(["Delay Risk Classification","Arrival Delay Regression"])
    with class_tab:
        if clf is None:st.error("Classification model file was not found.")
        else:
            with st.form("clf_form"):
                c1,c2,c3=st.columns(3)
                with c1:
                    fd=st.date_input("Flight Date",date.today())
                    airline=st.selectbox("Airline",sorted(df[AIRLINE].dropna().astype(str).unique()))
                with c2:
                    dep=st.time_input("Scheduled Departure",time(8,0))
                    duration=st.slider("Scheduled Flight Time (minutes)",30,600,120,5)
                with c3:
                    dist=st.slider("Distance (miles)",max(1,int(df.DISTANCE.min())),int(df.DISTANCE.max()),int(df.DISTANCE.median()),10,key="class_dist")
                submit=st.form_submit_button("Predict Delay Risk")
            if submit:
                try:
                    x=classification_row(clf,fd.month,fd.day,fd.isoweekday(),airline,dep.hour*100+dep.minute,duration,dist)
                    pred=int(clf.predict(x)[0]);prob=float(clf.predict_proba(x)[0,1]) if hasattr(clf,"predict_proba") else None
                    css,title=("bad","High Delay Risk") if pred else ("good","Likely On Time")
                    text=f"{prob*100:.1f}% estimated delay risk" if prob is not None else "Prediction completed"
                    st.markdown(f'<div class="{css}"><div class="result-title">{title}</div><div class="result-subtitle">{text}</div></div>',unsafe_allow_html=True)
                    if prob is not None:
                        g=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,number={"suffix":"%"},title={"text":"Estimated Delay Probability"},gauge={"axis":{"range":[0,100]},"steps":[{"range":[0,30],"color":"rgba(16,185,129,.25)"},{"range":[30,60],"color":"rgba(245,158,11,.25)"},{"range":[60,100],"color":"rgba(239,68,68,.25)"}]}))
                        st.plotly_chart(polish(g,360),use_container_width=True)
                except Exception as e:st.error(f"Prediction error: {e}")
    with reg_tab:
        if reg is None:st.error("Regression model file was not found.")
        else:
            with st.form("reg_form"):
                c1,c2=st.columns(2)
                with c1:
                    rd=st.date_input("Flight Date",date.today(),key="reg_date")
                    dep_delay=st.slider("Departure Delay (minutes)",-60,600,0)
                with c2:
                    rdist=st.slider("Distance (miles)",max(1,int(df.DISTANCE.min())),int(df.DISTANCE.max()),int(df.DISTANCE.median()),10,key="reg_dist")
                submit2=st.form_submit_button("Estimate Arrival Delay")
            if submit2:
                x=pd.DataFrame([[dep_delay,rdist,rd.month,rd.day,rd.isoweekday()]],columns=["DEPARTURE_DELAY","DISTANCE","MONTH","DAY","DAY_OF_WEEK"])
                try:
                    y=float(reg.predict(x)[0])
                    css,title=("bad","Expected Arrival Delay") if y>=15 else (("neutral","Minor Expected Delay") if y>0 else ("good","Expected On-Time Arrival"))
                    st.markdown(f'<div class="{css}"><div class="result-title">{title}</div><div class="result-subtitle">Predicted arrival delay: <strong>{y:.1f} minutes</strong></div></div>',unsafe_allow_html=True)
                except Exception as e:st.error(f"Prediction error: {e}")

with data_tab:
    st.markdown('<div class="section-title">Filtered Flight Records</div>',unsafe_allow_html=True)
    q=st.text_input("Search airline, airport or destination",placeholder="Example: Delta, Atlanta, LAX...")
    e=f.copy()
    if q:
        mask=pd.Series(False,index=e.index)
        for c in [AIRLINE,AIRPORT,"ORIGIN_AIRPORT","DESTINATION_AIRPORT"]:
            if c in e:mask|=e[c].astype(str).str.contains(q,case=False,na=False)
        e=e[mask]
    defaults=[c for c in [AIRLINE,AIRPORT,"DESTINATION_AIRPORT","MONTH","DAY","SCHEDULED_DEPARTURE","DISTANCE","DEPARTURE_DELAY","ARRIVAL_DELAY","IS_DELAYED"] if c in e]
    selected=st.multiselect("Columns to display",e.columns.tolist(),default=defaults)
    st.caption(f"{len(e):,} matching flights")
    if selected:
        st.dataframe(e[selected],use_container_width=True,hide_index=True,height=520)
        st.download_button("Download Filtered Data",e[selected].to_csv(index=False).encode("utf-8"),"filtered_flights.csv","text/csv",use_container_width=True)

st.markdown("---")
st.caption("Airline Delay Intelligence • Built with Streamlit, Plotly and XGBoost")
