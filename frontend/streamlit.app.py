import streamlit as st
import requests

# FastAPI endpoint
API_URL = "https://sample-cicd-gm27.onrender.com/predict"    # ← update this to your actual deployed URL

st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict house price using a Linear Regression model")

# User inputs
area = st.number_input(
    "Area (sqft)",
    min_value=300,
    max_value=5000,
    value=1200
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

if st.button("Predict Price"):
    payload = {
        "area": area,
        "bedrooms": bedrooms
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            price = result["predicted_price"]
            st.success(f"Estimated Price: ₹{price:,.2f}")
        else:
            st.error(f"Prediction failed. Status code: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Is it running?")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")