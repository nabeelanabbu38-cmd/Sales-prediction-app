import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Prediction Using ARIMA",
    page_icon="📊",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1>📊 Sales Prediction Using ARIMA Model</h1>
    <p>This application predicts future sales using past sales data and the ARIMA time-series model.</p>
    """,
    unsafe_allow_html=True
)

# ---------------- CSV FORMAT INFO ----------------
with st.expander("📌 CSV File Format (Important)"):
    st.markdown(
        """
        **Your CSV file must contain these columns:**

        - `Month` (YYYY-MM or date)
        - `Sales`

        **Example:**
        ```
        Month,Sales
        2024-01,120
        2024-02,150
        2024-03,170
        ```
        """
    )

# ---------------- INPUT SECTION ----------------
st.markdown("## ✍️ Enter Product & Sales Details")

col1, col2 = st.columns(2)

with col1:
    product_name = st.text_input("Product Name")
    product_price = st.number_input("Product Price", min_value=0.0)
    advertising_cost = st.number_input("Advertising Cost", min_value=0.0)
    promotion_cost = st.number_input("Promotion Cost", min_value=0.0)

with col2:
    st.markdown("### 📅 Enter Past 3 Months Sales")
    m1 = st.number_input("Month 1 Sales", min_value=0.0)
    m2 = st.number_input("Month 2 Sales", min_value=0.0)
    m3 = st.number_input("Month 3 Sales", min_value=0.0)

# ---------------- CSV UPLOAD (OPTIONAL) ----------------
st.markdown("## 📂 OR Upload CSV File")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# ---------------- DATA PREPARATION ----------------
sales_data = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    sales_data = df["Sales"].values
elif m1 > 0 and m2 > 0 and m3 > 0:
    sales_data = np.array([m1, m2, m3])

# ---------------- PREDICTION ----------------
if st.button("📈 Predict Future Sales"):

    if sales_data is None or len(sales_data) < 3:
        st.error("Please upload a CSV or enter past 3 months sales.")
    else:
        # ARIMA MODEL
        model = ARIMA(sales_data, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=3)

        # ---------------- OUTPUT ----------------
        st.markdown("## 📊 Prediction Results")

        left, right = st.columns(2)

        # -------- TABLE --------
        with left:
            result_df = pd.DataFrame({
                "Month": ["Next Month", "After 2 Months", "After 3 Months"],
                "Predicted Sales": forecast.round(2)
            })
            st.table(result_df)

        # -------- CHART --------
        with right:
            plt.figure()
            all_sales = np.concatenate([sales_data, forecast])
            plt.plot(all_sales, marker="o")
            plt.title("Sales Trend & Forecast")
            plt.xlabel("Time")
            plt.ylabel("Sales")
            st.pyplot(plt)

        # ---------------- SUMMARY ----------------
        st.markdown("## 🧾 Product Summary")
        st.write(f"**Product Name:** {product_name}")
        st.write(f"**Product Price:** {product_price}")
        st.write(f"**Advertising Cost:** {advertising_cost}")
        st.write(f"**Promotion Cost:** {promotion_cost}")
