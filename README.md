# 🌿 Plant Disease Detection Web App

A machine learning-based Streamlit web application that detects plant leaf diseases from images and provides helpful treatment suggestions via trusted sources.

---

## 🚀 Features

- 🌱 Upload a plant leaf image
- 🤖 Get disease predictions using a trained deep learning model
- 📊 View model confidence
- 🌐 Get instant treatment suggestions via:
  - [Google Search](https://www.google.com)
  - [Wikipedia](https://www.wikipedia.org)
  - [YouTube Videos](https://www.youtube.com)

---

## 🧠 Model Info

- Model Type: CNN (Convolutional Neural Network)
- Framework: TensorFlow / Keras
- Input Size: 224x224 RGB images
- Output: Predicted class (disease name)

---

## 📂 Files in This Repository

- `main.py` – Streamlit app script
- `class_indices.json` – Dictionary mapping class labels
- `requirements.txt` – Dependencies for running the app
- **Model file (`.h5`) is stored in Google Drive due to size limit**

---

## 🔗 Model File Download

Since the model file is too large for GitHub, it's stored on Google Drive.  
🔗 [Download model from Google Drive](https://drive.google.com/file/d/YOUR_MODEL_ID/view?usp=sharing)

> (Update the link above with your actual shared model link)

---

## ▶️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/plant-disease-detector.git
   cd plant-disease-detector
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run main.py

Streamlit app link :https://plant-disease-prediction-suggestion.streamlit.app/

🙋‍♀️ Author
Vaishnavi Deshmukh –(https://www.linkedin.com/in/vaishnavi-deshmukh-004927282?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
