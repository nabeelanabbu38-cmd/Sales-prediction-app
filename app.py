import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📈 Sales Prediction App",
    page_icon="📊",
    layout="centered"
)

st.title("📈 Sales Prediction using ARIMA")
st.subheader("Time Series Forecasting Web App")

st.markdown(
    """
    This application predicts **future sales** using the **ARIMA time-series model**.
    
    **Steps:**
    1. Upload sales dataset  
    2. Select date & sales columns  
    3. Train ARIMA model  
    4. Forecast future sales  
    """
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV file (Date, Sales)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully")

    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- COLUMN SELECTION ----------------
    date_col = st.selectbox("📅 Select Date Column", df.columns)
    sales_col = st.selectbox("💰 Select Sales Column", df.columns)

    # ---------------- DATA PREPROCESSING ----------------
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df.set_index(date_col, inplace=True)

    sales_data = df[sales_col]

    st.write("### 📊 Historical Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(sales_data, label="Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    # ---------------- MODEL PARAMETERS ----------------
    st.write("### ⚙️ ARIMA Model Configuration")

    p = st.slider("AR (p)", 0, 5, 1)
    d = st.slider("Differencing (d)", 0, 2, 1)
    q = st.slider("MA (q)", 0, 5, 1)

    forecast_days = st.slider(
        "📆 Forecast Days",
        min_value=1,
        max_value=60,
        value=10
    )

    # ---------------- TRAIN & PREDICT ----------------
    if st.button("🚀 Train Model & Predict"):
        try:
            model = ARIMA(sales_data, order=(p, d, q))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=forecast_days)

            future_dates = [
                sales_data.index[-1] + timedelta(days=i)
                for i in range(1, forecast_days + 1)
            ]

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Sales": forecast.values
            })

            st.success("✅ Forecast Generated Successfully")

            st.write("### 🔮 Predicted Sales")
            st.dataframe(forecast_df)

            # ---------------- FORECAST PLOT ----------------
            st.write("### 📉 Sales Forecast Visualization")
            fig2, ax2 = plt.subplots()
            ax2.plot(sales_data, label="Historical Sales")
            ax2.plot(
                forecast_df["Date"],
                forecast_df["Predicted Sales"],
                label="Predicted Sales",
                linestyle="--"
            )
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Sales")
            ax2.legend()
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    st.info("👆 Please upload a CSV file to get started")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "👩‍💻 **Project:** Sales Prediction using ARIMA  \n"
    "📌 **Tech Stack:** Python, Streamlit, ARIMA, Pandas, Matplotlib"
    )
