import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TensorFlow/Keras model
model = tf.keras.models.load_model('teeth_model.h5')

# Define class labels
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

st.title('Teeth Classification App')
st.write('Upload an image of a tooth to classify it.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Open the image and ensure it's in RGB mode
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.write(f"Prediction: {predicted_class}")
