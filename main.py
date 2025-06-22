import os
import json
import urllib.parse
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Set up Streamlit page
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")

# App title and description
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Plant Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a leaf image to detect disease and get treatment suggestions.</p>", unsafe_allow_html=True)
st.markdown("---")

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_detection_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load model and class indices
model = tf.keras.models.load_model(model_path, compile=False)
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Image preprocessing function
def load_and_preprocess_image_pil(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image)

    # Fix for grayscale or 4-channel images
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Prediction function
def predict_image_class(image):
    preprocessed_img = load_and_preprocess_image_pil(image)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_name = class_indices.get(str(predicted_index), "Unknown")
    confidence = predictions[0][predicted_index]
    return predicted_name, confidence

# Upload section
uploaded_image = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        display_image = image.resize((300, 300))  # For display only

        st.image(display_image, caption="🖼️ Uploaded Image", use_container_width=False)

        if st.button("🔍 Classify"):
            predicted_class, confidence = predict_image_class(image)
            st.session_state["prediction"] = predicted_class
            st.session_state["confidence"] = confidence
            st.session_state["classified"] = True

    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        st.stop()

# Show prediction result
if st.session_state.get("classified", False):
    prediction = st.session_state["prediction"]
    confidence = st.session_state["confidence"]

    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; background-color: #f4fff2;">
        <h3>✅ Disease Detected: <span style='color:#d14;'> {prediction}</span></h3>
        <p>📊 Model Confidence: <strong>{confidence * 100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧪 Need Help? Get Treatment Suggestions:")

    if st.button("🌐 Show Treatment Resources"):
        query = prediction.replace("___", " ").replace("_", " ")
        encoded_query = urllib.parse.quote(query + " treatment for plants")

        google_url = f"https://www.google.com/search?q={encoded_query}"
        wikipedia_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        youtube_url = f"https://www.youtube.com/results?search_query={encoded_query}+treatment"

        st.markdown("### 🌐 Suggested Resources")
        st.markdown(f"🔎 [Google Search]({google_url})", unsafe_allow_html=True)
        st.markdown(f"📘 [Wikipedia Article]({wikipedia_url})", unsafe_allow_html=True)
        st.markdown(f"🎥 [YouTube Videos]({youtube_url})", unsafe_allow_html=True)

# Reset session
st.markdown("---")
if st.button("🔄 Reset"):
    st.session_state.clear()
    st.success("Session cleared. Please refresh the page manually.")
