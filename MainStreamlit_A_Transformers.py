import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Memuat model
model = load_model(r'D:\TUBES PMDPM\mobilenet\model_mobilenet.h5')
class_names = ['Busuk', 'Matang', 'Mentah']

def classify_image(image_path):
    try:
        # Memuat gambar dan mengubah ukurannya sesuai dengan model
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        # Prediksi menggunakan model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  # Softmax untuk mendapatkan probabilitas

        # Mendapatkan kelas dengan confidence tertinggi
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

def display_progress_bar(confidence, class_names):
    for i, class_name in enumerate(class_names):
        percentage = confidence[i] * 100
        st.sidebar.progress(int(percentage))  # Streamlit progress bar
        st.sidebar.write(f"{class_name}: {percentage:.2f}%")

# Streamlit UI
st.title("Prediksi Kematangan Buah Rambutan")

uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan) dan pastikan gambar sudah terupload dengan benar, dan jika terjadi kesalahan refresh saja halamannya. terima kasih.", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            label, confidence = classify_image(uploaded_file.name)
            
            if label != "Error":
                st.sidebar.write(f"*Prediksi: {label}*")
                st.sidebar.write("Confidence:")
                display_progress_bar(confidence, class_names)
                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silahkan unggah setidaknya satu gambar untuk diprediksi")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)