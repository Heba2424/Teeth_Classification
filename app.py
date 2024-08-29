import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  # Use Keras load_model for .h5 files
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

# Ensure you use the correct path format for Windows
model_path = r"teeth_model.keras"

# Load the model using Keras's load_model function
model = load_model(model_path)

# Define the labels based on your encoder
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

st.title('Teeth Classification App')
st.write('Upload an image of a tooth to classify it.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Open the image and ensure it's in RGB mode
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((224, 224))  # Ensure this matches your model's expected input size
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Preprocessing if required
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Prediction: {predicted_class}")
