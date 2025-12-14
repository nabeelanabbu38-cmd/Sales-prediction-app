import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Sales Prediction Using ARIMA",
    layout="wide"
)

# ------------------ TITLE ------------------
st.markdown("## 📊 Sales Prediction Using ARIMA Model")
st.write(
    "This application predicts **future sales** using historical sales data "
    "with the **ARIMA time-series forecasting model**."
)

# ------------------ CSV FORMAT INFO ------------------
with st.expander("📌 CSV File Format (Important)"):
    st.markdown("""
    Your CSV **must contain** the following columns:

    | Column Name | Description |
    |------------|-------------|
    | date | Month or date (YYYY-MM or YYYY-MM-DD) |
    | sales | Sales value (numeric) |

    **Example:**
    ```
    date,sales
    2024-01,1200
    2024-02,1500
    2024-03,1700
    ```
    """)

# ------------------ USER INPUT SECTION ------------------
st.markdown("### 🧾 Product & Marketing Details")

col1, col2, col3, col4 = st.columns(4)

with col1:
    product_name = st.text_input("Product Name")

with col2:
    product_price = st.number_input("Product Price", min_value=0.0)

with col3:
    advertising_cost = st.number_input("Advertising Cost", min_value=0.0)

with col4:
    promotion_cost = st.number_input("Promotion Cost", min_value=0.0)

# ------------------ DATA INPUT OPTIONS ------------------
st.markdown("### 📥 Upload CSV or Enter Past Sales Manually")

data_option = st.radio(
    "Choose input method:",
    ["Upload CSV", "Enter Manually"]
)

df = None

# ------------------ CSV UPLOAD ------------------
if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

# ------------------ MANUAL INPUT ------------------
if data_option == "Enter Manually":
    st.write("Enter **past 3 months sales data**")

    m1 = st.number_input("Month 1 Sales", min_value=0)
    m2 = st.number_input("Month 2 Sales", min_value=0)
    m3 = st.number_input("Month 3 Sales", min_value=0)

    if st.button("Create Dataset"):
        df = pd.DataFrame({
            "date": pd.date_range(start="2024-01-01", periods=3, freq="M"),
            "sales": [m1, m2, m3]
        })

# ------------------ PREDICTION ------------------
if df is not None and len(df) >= 3:

    st.markdown("## 📈 Sales Analysis & Prediction")

    colA, colB = st.columns(2)

    # ------------------ HISTORICAL CHART ------------------
    with colA:
        st.subheader("📊 Historical Sales")
        fig1, ax1 = plt.subplots()
        ax1.plot(df["date"], df["sales"], marker="o")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Sales")
        st.pyplot(fig1)

    # ------------------ ARIMA MODEL ------------------
    model = ARIMA(df["sales"], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=3)

    future_dates = pd.date_range(
        start=df["date"].iloc[-1], periods=4, freq="M"
    )[1:]

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "sales": forecast
    })

    # ------------------ FORECAST CHART ------------------
    with colB:
        st.subheader("🔮 Future Sales Prediction")
        fig2, ax2 = plt.subplots()
        ax2.plot(df["date"], df["sales"], label="Past Sales", marker="o")
        ax2.plot(forecast_df["date"], forecast_df["sales"],
                 label="Predicted Sales", marker="o")
        ax2.legend()
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Sales")
        st.pyplot(fig2)

    # ------------------ TABLE ------------------
    st.markdown("### 📋 Predicted Sales Table")
    st.dataframe(forecast_df)

    # ------------------ SUMMARY ------------------
    st.success(f"""
    ✅ Prediction completed for **{product_name}**

    • Product Price: ₹{product_price}  
    • Advertising Cost: ₹{advertising_cost}  
    • Promotion Cost: ₹{promotion_cost}  
    """)

else:
    st.info("Please upload a CSV or enter past sales data to continue.")
