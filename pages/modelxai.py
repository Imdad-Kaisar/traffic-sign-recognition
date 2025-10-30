import streamlit as st
import numpy as np

from src.model_loader import load_model, load_class_names, load_metrics
from src.preprocess import load_and_preprocess_image
from src.xai import grad_cam

st.title("📊 Model Performance & Explainable AI")

# Show metrics
metrics = load_metrics()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Precision", f"{metrics['precision_macro']*100:.2f}%")
col3.metric("Recall", f"{metrics['recall_macro']*100:.2f}%")
col4.metric("F1 Score", f"{metrics['f1_macro']*100:.2f}%")

col5, col6, col7 = st.columns(3)
col5.metric("Jaccard", f"{metrics['jaccard_macro']:.4f}")
col6.metric("MCC", f"{metrics['mcc']:.3f}")
col7.metric("ROC-AUC", f"{metrics['roc_auc_micro']:.3f}")

st.write("---")
st.subheader("🧠 Grad-CAM Explanation for Traffic Sign")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    img, batch = load_and_preprocess_image(file)

    # Run Grad-CAM
    heatmap, idx, preds = grad_cam(load_model(), batch)
    class_names = load_class_names()

    label = class_names[idx]
    confidence = preds[idx]  # ✅ FIXED (1-D array indexing)

    st.write(f"### ✅ Prediction: **{label}**")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Build overlay
    overlay = np.array(img).astype("float32") / 255
    cam = np.array(heatmap).astype("float32") / 255
    cam = np.expand_dims(cam, axis=-1)

    # Grad-CAM colormap (Blue→Red)
    heat = np.concatenate([cam, np.zeros_like(cam), 1 - cam], axis=-1)
    blend = (0.6 * overlay + 0.4 * heat)
    blend = np.clip(blend, 0, 1)
    blend_img = (blend * 255).astype("uint8")

    colA, colB, colC = st.columns(3)
    colA.image(img, caption="Original", use_column_width=True)
    colB.image(heatmap, caption="Heatmap", use_column_width=True)
    colC.image(blend_img, caption="Overlay", use_column_width=True)
else:
    st.info("Upload a traffic sign image to visualize Grad-CAM.")
