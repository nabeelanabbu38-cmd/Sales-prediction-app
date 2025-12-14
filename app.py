import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="wide"
)

# ------------------ TITLE ------------------
st.title("📊 Sales Prediction Using ARIMA Model")
st.write(
    "This application predicts future sales using historical sales data "
    "with the **ARIMA time series forecasting model**."
)

# ------------------ SIDEBAR ------------------
st.sidebar.header("📂 Upload or Manual Input")

# ------------------ CSV FORMAT INFO ------------------
with st.expander("📌 CSV File Format (Important)"):
    st.markdown("""
    Your CSV file **must contain** the following columns:

    | Column Name | Description |
    |------------|-------------|
    | `Date` | Date of sale (YYYY-MM-DD format) |
    | `Sales` | Sales value (numeric) |

    **Example:**
    ```
    Date,Sales
    2023-01-01,1200
    2023-01-02,1350
    2023-01-03,1100
    ```
    """)

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload Sales CSV",
    type=["csv"]
)

# ------------------ MANUAL INPUT ------------------
st.sidebar.subheader("✍️ Manual Sales Entry")
manual_date = st.sidebar.date_input("Date")
manual_sales = st.sidebar.number_input(
    "Sales Value",
    min_value=0.0,
    step=100.0
)

# ------------------ DATAFRAME INIT ------------------
data = None

# ------------------ LOAD CSV ------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")

# ------------------ MANUAL ADD ------------------
if data is not None and st.sidebar.button("➕ Add Manual Entry"):
    new_row = pd.DataFrame({
        "Date": [manual_date],
        "Sales": [manual_sales]
    })
    data = pd.concat([data, new_row], ignore_index=True)
    data = data.sort_values("Date")

# ------------------ MAIN CONTENT ------------------
if data is not None:

    col1, col2 = st.columns(2)

    # ---------- RAW DATA ----------
    with col1:
        st.subheader("📋 Sales Data")
        st.dataframe(data, use_container_width=True)

    # ---------- SALES TREND ----------
    with col2:
        st.subheader("📈 Sales Trend")
        fig, ax = plt.subplots()
        ax.plot(data["Date"], data["Sales"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title("Historical Sales Trend")
        st.pyplot(fig)

    # ---------- ARIMA SETTINGS ----------
    st.subheader("⚙️ Forecast Settings")
    forecast_days = st.slider(
        "Select number of days to predict",
        min_value=1,
        max_value=60,
        value=7
    )

    # ---------- ARIMA MODEL ----------
    try:
        model = ARIMA(data["Sales"], order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_days)

        future_dates = pd.date_range(
            start=data["Date"].iloc[-1] + pd.Timedelta(days=1),
            periods=forecast_days
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Sales": forecast
        })

        # ---------- PREDICTION OUTPUT ----------
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🔮 Predicted Sales")
            st.dataframe(forecast_df, use_container_width=True)

        with col4:
            st.subheader("📊 Actual vs Predicted")
            fig2, ax2 = plt.subplots()
            ax2.plot(data["Date"], data["Sales"], label="Actual Sales")
            ax2.plot(future_dates, forecast, label="Predicted Sales", linestyle="--")
            ax2.legend()
            ax2.set_title("Sales Forecast")
            st.pyplot(fig2)

    except Exception as e:
        st.error("❌ Error in ARIMA model. Please check your data format.")

else:
    st.info("👈 Upload a CSV file or add data manually to begin prediction.")
