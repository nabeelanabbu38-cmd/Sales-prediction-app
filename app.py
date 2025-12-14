import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Prediction using ARIMA",
    page_icon="📊",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📊 Sales Prediction Using ARIMA Model")
st.write(
    "This application predicts future sales using historical sales data "
    "with the **ARIMA time series forecasting model**."
)

# ---------------- CSV FORMAT INFO ----------------
with st.expander("📌 CSV File Format (Important)"):
    st.markdown("""
    Your CSV file **must contain exactly two columns**:

    | Column Name | Description |
    |------------|-------------|
    | `Date`     | Date (YYYY-MM-DD) |
    | `Sales`    | Sales value (number) |

    **Example:**
    ```
    Date,Sales
    2023-01-01,120
    2023-02-01,150
    2023-03-01,180
    ```

    - Date column must be in **date format**
    - Sales must be **numeric**
    """)

st.info("👉 Upload a CSV file OR add data manually to begin prediction.")

# ---------------- DATA INPUT SECTION ----------------
tab1, tab2 = st.tabs(["📂 Upload CSV", "✍️ Manual Data Entry"])

data = None

# ---------- TAB 1: CSV UPLOAD ----------
with tab1:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date")

        st.success("✅ CSV file loaded successfully")
        st.dataframe(data)

# ---------- TAB 2: MANUAL ENTRY ----------
with tab2:
    st.write("Enter sales data manually:")

    dates = st.text_area(
        "Dates (comma separated, YYYY-MM-DD)",
        "2023-01-01,2023-02-01,2023-03-01"
    )
    sales = st.text_area(
        "Sales values (comma separated)",
        "100,150,180"
    )

    if st.button("Add Manual Data"):
        try:
            date_list = [d.strip() for d in dates.split(",")]
            sales_list = [float(s.strip()) for s in sales.split(",")]

            data = pd.DataFrame({
                "Date": pd.to_datetime(date_list),
                "Sales": sales_list
            }).sort_values("Date")

            st.success("✅ Manual data added successfully")
            st.dataframe(data)

        except Exception as e:
            st.error("❌ Error in manual data format. Please check inputs.")

# ---------------- MODEL & VISUALIZATION ----------------
if data is not None and len(data) >= 5:
    st.subheader("📈 Sales Analysis & Forecast")

    data.set_index("Date", inplace=True)

    # Forecast horizon
    steps = st.slider("Select number of months to predict", 1, 12, 5)

    # Train ARIMA
    model = ARIMA(data["Sales"], order=(1, 1, 1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    forecast_index = pd.date_range(
        start=data.index[-1],
        periods=steps + 1,
        freq="M"
    )[1:]

    forecast_series = pd.Series(forecast, index=forecast_index)

    # ---------------- SIDE BY SIDE CHARTS ----------------
    col1, col2 = st.columns(2)

    # ---- Chart 1: Historical Sales ----
    with col1:
        st.markdown("### 📊 Historical Sales")
        fig1, ax1 = plt.subplots()
        ax1.plot(data.index, data["Sales"], marker="o")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Sales")
        st.pyplot(fig1)

    # ---- Chart 2: Forecast ----
    with col2:
        st.markdown("### 🔮 Forecasted Sales")
        fig2, ax2 = plt.subplots()
        ax2.plot(data.index, data["Sales"], label="Actual")
        ax2.plot(forecast_series.index, forecast_series, label="Forecast", marker="o")
        ax2.legend()
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Sales")
        st.pyplot(fig2)

    # ---------------- FORECAST TABLE ----------------
    st.subheader("📋 Forecasted Values")
    st.dataframe(forecast_series.reset_index().rename(
        columns={"index": "Date", 0: "Predicted Sales"}
    ))

else:
    st.warning("⚠️ Please provide at least **5 data points** for ARIMA prediction.")
