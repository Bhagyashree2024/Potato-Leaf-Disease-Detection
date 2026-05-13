import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "potato_model.h5",
    compile=False,
    safe_mode=False
)

# Disease classes
classes = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# App title
st.title("🥔 Potato Leaf Disease Detection")

st.write("Upload a potato leaf image")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    # Resize image
    img = image.resize((128, 128))

    # Convert image to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    result = classes[predicted_class]

    st.success(f"Prediction: {result}")