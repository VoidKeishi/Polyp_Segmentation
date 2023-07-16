import streamlit as st
from Model import load_model
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib import request
from keras.preprocessing import image_dataset_from_directory

@st.cache_resource
def load_unet():
    url = 'https://github.com/VoidKeishi/Polyp_Segmentation/releases/download/v1.0/unet_model_weight_gg.h5'
    if 'unet_model_weight_gg.h5' not in os.listdir('Weights'):
        print('Downloading...')
        request.urlretrieve(url, 'Weights/unet_model_weight_gg.h5')
    model = load_model()
    return model

# Load the U-Net model
model = load_unet()

# Set Streamlit title and description
st.title('Polyp Segmentation')
st.write('This app performs polyp segmentation using a U-Net model.')

# Add a file uploader to allow the user to upload an image
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

# Check if the user has uploaded a file
if uploaded_file is not None:
    # Read the uploaded image file
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((128, 128))  # Resize the image to match the model's input shape
    image = np.array(image.convert('RGB'))

    # Preprocess the image
    image = image / 255.0  # Normalize the pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add an extra dimension to match the model's input shape

    # Perform polyp segmentation using the loaded U-Net model
    segmented_image = model.predict(image)

    # Convert segmented image to grayscale
    segmented_image_gray = np.squeeze(segmented_image)  # Remove batch dimension if present
    segmented_image_gray = np.uint8(segmented_image_gray * 255)

    # Create a binary black and white view by rounding the pixel values
    binary_image = np.round(segmented_image)

    # Convert binary image to PIL Image
    binary_image_pil = Image.fromarray(np.squeeze(binary_image).astype('uint8') * 255).convert('1')

    # Calculate the heat map
    heatmap = segmented_image_gray.copy()

    # Create a color map for the heat map
    cmap = plt.cm.jet

    # Plot the heat map overlay
    fig, ax = plt.subplots()
    ax.imshow(image[0], cmap='gray')
    ax.imshow(heatmap, cmap=cmap, alpha=0.5)
    ax.axis('off')
    plt.tight_layout()

    # Display the images and text
    c1, c2, c3, c4 = st.columns(4)

    c1.image(image[0], channels='RGB')
    c1.text('Input Image')

    c2.image(segmented_image[0], channels='L')
    c2.text('Segmented Image')

    c3.image(binary_image_pil, channels='L')
    c3.text('Binary Image')

    c4.pyplot(fig)
