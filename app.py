import streamlit as st

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="wide")

st.title("🚦 Traffic Sign Recognition")
st.write(
"""
Upload a traffic sign and the model will classify it.
Use the sidebar to navigate:

- 🔮 Predict
- 📊 Model & XAI
"""
)
