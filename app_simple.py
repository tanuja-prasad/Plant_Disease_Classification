import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Plant Leaf Disease Recognition",
    page_icon="🌿",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "plant_disease_cnn_fixed.keras",
        compile=False
    )

model = load_model()


class_names = [
    "tomato_healthy",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold"
]


st.title("Plant Leaf Disease Recognition")

st.write(
    "Upload a tomato leaf image to predict one of four categories."
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Resize image for CNN
    resized_image = image.resize((128, 128))

    # Convert image to NumPy array
    img_array = np.array(resized_image, dtype=np.float32)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    # Prediction result
    st.markdown("### Prediction Result")

    col1, col2 = st.columns([1, 3], gap="medium")

    with col1:
        st.image(image, width=160)

    with col2:
        disease_name = class_names[predicted_index].replace("_", " ").title()

        st.markdown(f"**Predicted disease: {disease_name}**")

        st.markdown(
            f"<h2 style='color:#16a34a;margin:0;'>{confidence:.1f}%</h2>",
            unsafe_allow_html=True
        )

        st.caption("Model confidence for the predicted class")

    st.divider()

    # Confidence scores
    st.markdown(f"**Model confidence: {confidence:.1f}%**")

    for i, name in enumerate(class_names):
        st.write(name)
        st.progress(float(predictions[i]))