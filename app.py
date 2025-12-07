import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Beautiful UI Setup
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="📈",
    layout="wide"
)

# Title Banner
st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>📈 Multi-Product Sales Forecasting App</h1>
    <p style='text-align:center; font-size:18px;'>Forecast future sales using ARIMA model with a beautiful interface</p>
""", unsafe_allow_html=True)

# Tabs in UI
tabs = st.tabs(["🏠 Home", "📥 Upload Data", "📈 Forecast", "📊 Charts", "🔧 Model Info"])

# Global store for dataset
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# -------------------------------------------
# HOME TAB
# -------------------------------------------
with tabs[0]:
    st.markdown("### Welcome to the Multi-Product Sales Forecasting App!")
    st.write("""
    This app allows you to:
    - Upload your sales dataset  
    - Select a product  
    - Forecast future monthly sales using ARIMA  
    - View charts and download predictions  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=200)

# -------------------------------------------
# UPLOAD DATA TAB
# -------------------------------------------
with tabs[1]:
    st.header("📥 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Store in session
        st.session_state.dataset = df
        
        st.success("File uploaded successfully!")
        st.write("### 🔍 Preview of Dataset")
        st.dataframe(df.head())

        # Requirements check
        required_cols = ["date", "product", "sales"]
        if not all(col in df.columns for col in required_cols):
            st.error("Dataset must contain columns: date, product, sales")
        else:
            st.success("Dataset format is valid!")


# -------------------------------------------
# FORECAST TAB (Manual Data Entry)
# -------------------------------------------
with tabs[2]:
    st.header("📈 Forecast Future Sales (Manual Input)")

    st.info("Enter past sales data manually to generate ARIMA forecast.")

    # Select product
    selected_product = st.text_input("Enter Product Name", "Product A")

    # Number of past months
    n_past = st.number_input("How many past months of data do you have?", min_value=1, max_value=36, value=6)

    st.write("### Enter Past Sales Data")

    # Create input boxes dynamically
    past_sales = []
    for i in range(n_past):
        value = st.number_input(f"Month {i+1} sales", min_value=0.0, value=100.0)
        past_sales.append(value)

    # Convert to DataFrame
    import pandas as pd
    import numpy as np

    # Create date index automatically
    from datetime import datetime
    import pandas as pd

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_past, freq="M")

    df_manual = pd.DataFrame({
        "date": dates,
        "product": [selected_product] * n_past,
        "sales": past_sales
    })

    st.write("### 📊 Your Entered Data")
    st.dataframe(df_manual)

    # Number of months to forecast
    n_future = st.slider("Months to Forecast", 1, 24, 12)

    # Train ARIMA Model
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(df_manual["sales"], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=n_future)

        st.success("ARIMA Model Trained Successfully!")
    except Exception as e:
        st.error(f"Error training ARIMA model: {e}")
        st.stop()

    # Create future date index
    future_dates = pd.date_range(start=df_manual["date"].iloc[-1] + pd.offsets.MonthEnd(1),
                                 periods=n_future, freq="M")

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Predicted Sales": forecast
    })

    st.write("### 🔮 Forecasted Sales")
    st.dataframe(forecast_df)

    # Plot graph
    import matplotlib.pyplot as plt

    st.write("### 📈 Forecast Graph")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_manual["date"], df_manual["sales"], label="Historical Sales")
    ax.plot(future_dates, forecast, label="Forecasted Sales")
    ax.legend()
    st.pyplot(fig)

    # Download CSV
    csv_data = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Forecast CSV", csv_data, "forecast.csv", "text/csv")
    
# -------------------------------------------
# CHARTS TAB
# -------------------------------------------
with tabs[3]:
    st.header("📊 Visual Insights")
    if st.session_state.dataset is None:
        st.warning("Upload data to view charts.")
    else:
        df = st.session_state.dataset

        # Sales over time
        st.subheader("📌 Sales Trend for All Products")
        fig, ax = plt.subplots(figsize=(10, 5))

        for product in df["product"].unique():
            temp = df[df["product"] == product]
            ax.plot(temp["date"], temp["sales"], label=product)

        ax.legend()
        st.pyplot(fig)

# -------------------------------------------
# MODEL INFO TAB
# -------------------------------------------
with tabs[4]:
    st.header("🔧 Model Details")
    st.write("""
        - **Model Used:** ARIMA (Auto Regressive Integrated Moving Average)  
        - **Best For:** Time-series forecasting  
        - **Supports:** Trend + seasonality  
        - **Input Required:** date, product, sales  
    """)
