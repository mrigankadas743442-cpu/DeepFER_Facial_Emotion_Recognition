import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("😊 DeepFER - Facial Emotion Recognition")
st.write("Upload a facial image to predict the emotion.")

model = tf.saved_model.load("EmotionSavedModel")
predict_fn = model.signatures["serve"]

labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("L")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((48, 48))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))

    output = predict_fn(tf.constant(img))
    prediction = list(output.values())[0].numpy()[0]

    emotion = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Emotion: {emotion}")
    st.metric("Confidence", f"{confidence:.2f}%")
