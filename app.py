import streamlit as st
import onnxruntime as rt
import numpy as np
from PIL import Image
import requests

# Function to download the model from Dropbox
def download_model(dropbox_url, output_path):
    # Convert Dropbox URL to direct download link
    direct_url = dropbox_url.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "")
    
    # Download the file
    r = requests.get(direct_url)
    with open(output_path, 'wb') as f:
        f.write(r.content)

# Download the model from Dropbox
dropbox_url = "https://www.dropbox.com/scl/fi/t2s43bzy00kk3lci70so3/model.onnx?rlkey=jrfke2at0vr6n7r9g5i7starv&st=jt2tyy9a&dl=0"  # Replace with your Dropbox model URL
model_path = "model.onnx"
download_model(dropbox_url, model_path)

# Load the ONNX model
sess = rt.InferenceSession(model_path)

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
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using ONNX runtime
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: img_array})
    predicted_class = class_names[np.argmax(pred_onx[0])]

    st.write(f"Prediction: {predicted_class}")
