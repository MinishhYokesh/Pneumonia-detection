import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import os

# Direct download link for the model
model_url = 'https://drive.google.com/uc?id=1AuMUY0N6-MjLPFxdpm35A61dwS1EUVWl'
model_path = 'model.h5'

# Function to download and save the model
def download_model(url, file_path='model.h5'):
    if not os.path.exists(file_path):
        st.write("Downloading model...")
        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        st.success('Model downloaded successfully!')
    else:
        st.write("Model already downloaded.")

# Download the model when the app runs
download_model(model_url, model_path)

# Load the model
model = tf.keras.models.load_model(model_path)
model = tf.keras.models.load_model(model_path, compile=False)

# Define class labels
class_labels = ['Bacteria Pneumonia', 'normal', 'Viral Pneumonia']

# Streamlit UI
st.title("Pneumonia Detection Dashboard")
uploaded_file = st.file_uploader("Upload your X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':  # Convert grayscale to RGB if necessary
        image = image.convert('RGB')
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image for prediction
    image = image.resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(input_arr)
    predicted_class = np.argmax(prediction)

    # Display the prediction
    st.subheader(f'Prediction: {class_labels[predicted_class]}')

