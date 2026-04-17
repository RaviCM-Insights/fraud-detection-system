import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Fraud Detection System")

st.write("Enter all transaction features separated by commas.")
st.write("Example: 0.1, -1.2, 0.5, ... (same number of features as training data)")

# Input field
features = st.text_input("Enter all features (comma separated)")

if st.button("Predict"):
    try:
        # Convert input string to list of floats
        values = list(map(float, features.split(",")))

        # Convert to numpy array
        data = np.array([values])

        # Scale input
        data = scaler.transform(data)

        # Predict
        prediction = model.predict(data)

        # Output result
        if prediction[0] == 1:
            st.error("🚨 Fraud Transaction Detected!")
        else:
            st.success("✅ Safe Transaction")

    except ValueError:
        st.error("❌ Please enter valid numeric values separated by commas.")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")