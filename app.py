import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("sales_model.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #1c1f26;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    .metric-box {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("📊 Sales Forecast Dashboard")
st.markdown("### Predict future sales using machine learning")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("📥 Input Recent Sales Data")

col1, col2, col3 = st.columns(3)

with col1:
    lag_1 = st.number_input("Yesterday's Sales", min_value=0.0)
    lag_2 = st.number_input("2 Days Ago", min_value=0.0)

with col2:
    lag_7 = st.number_input("Same Day Last Week", min_value=0.0)
    rolling_mean_3 = st.number_input("3-Day Avg", min_value=0.0)

with col3:
    rolling_mean_7 = st.number_input("7-Day Avg", min_value=0.0)
    day_of_week = st.selectbox("Day of Week", 
                              ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    month = st.selectbox("Month", list(range(1,13)))

# Convert day to numeric
day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
day_of_week = day_map[day_of_week]

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Sales", use_container_width=True):

    input_data = pd.DataFrame([[
        lag_1, lag_2, lag_7,
        rolling_mean_3, rolling_mean_7,
        day_of_week, month
    ]], columns=[
        'lag_1','lag_2','lag_7',
        'rolling_mean_3','rolling_mean_7',
        'day_of_week','month'
    ])

    prediction = model.predict(input_data)[0]

    # ---------------- OUTPUT ----------------
    st.subheader("📈 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            label="Predicted Sales",
            value=f"${prediction:,.2f}"
        )

    with colB:
        change = prediction - lag_1
        st.metric(
            label="Change vs Yesterday",
            value=f"{change:.2f}",
            delta=f"{(change/lag_1*100) if lag_1!=0 else 0:.2f}%"
        )

    # ---------------- VISUAL ----------------
    st.markdown("### 📊 Trend Visualization")

    trend_data = pd.DataFrame({
        "Values": [lag_2, lag_1, prediction]
    }, index=["2 Days Ago", "Yesterday", "Predicted"])

    st.line_chart(trend_data)

    st.markdown("---")
    st.success("Prediction generated using trained ML model")