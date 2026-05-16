# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.gradcam.gradcam import make_gradcam_heatmap
from src.gradcam.display import overlay_gradcam

MODEL_PATH = "model/model.keras"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁")
st.title("Pneumonia Detector")
st.write("Upload a Chest X-ray to detect Pneumonia with Grad-CAM visualization.")

uploaded = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded:
    model = load_model()

    img = load_img(uploaded, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_batch = np.expand_dims(preprocess_input(img_array), axis=0)

    prediction = model.predict(img_batch)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original X-ray", use_container_width=True)
    with col2:
        heatmap = make_gradcam_heatmap(img_batch, model)
        gradcam_img = overlay_gradcam(img_array / 255.0, heatmap)
        st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)

    color = "red" if label == "PNEUMONIA" else "green"
    st.markdown(f"### Prediction: :{color}[{label}]")
    st.write(f"Confidence: `{confidence:.4f}`")
