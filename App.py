import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Load Model & Vectorizer (Pastikan filenya sudah kamu upload ke GitHub)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# 2. Fungsi Pembersihan Teks
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stemmer.stem(text)
    return text

# 3. Tampilan Streamlit
st.title("📊 Aplikasi Analisis Sentimen")
st.write("Masukkan kalimat untuk dianalisis:")

user_input = st.text_area("Input teks di sini...")

if st.button("Analisis"):
    if user_input:
        # Proses Prediksi
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        # Output Hasil (Label -1: Negatif, 0: Netral, 1: Positif)
        label_map = {1: "Positif", 0: "Netral", -1: "Negatif"}
        hasil = label_map.get(prediction, "Tidak Diketahui")
        
        if hasil == "Positif":
            st.success(f"Sentimen: {hasil} 😊")
        elif hasil == "Negatif":
            st.error(f"Sentimen: {hasil} 😡")
        else:
            st.info(f"Sentimen: {hasil} 😐")
    else:
        st.warning("Mohon isi teksnya dulu!")
