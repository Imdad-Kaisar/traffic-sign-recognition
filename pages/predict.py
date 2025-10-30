import streamlit as st
from src.model_loader import load_model, load_class_names
from src.preprocess import load_and_preprocess_image
from src.predict import top_k_predictions

st.title("🔮 Predict Traffic Sign")

file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if file:
    img, batch = load_and_preprocess_image(file)
    st.image(img, caption="Uploaded Image", width=300)

    model = load_model()
    class_names = load_class_names()

    preds = model.predict(batch, verbose=0)
    top5 = top_k_predictions(preds, class_names)

    st.subheader("Prediction Results")
    st.metric("Top Prediction", top5[0][0], f"{top5[0][1]*100:.2f}%")

    st.write("### Top-5 classes")
    for label, prob in top5:
        st.write(f"- {label}: **{prob*100:.2f}%**")
else:
    st.info("Upload a traffic sign image to classify it.")
