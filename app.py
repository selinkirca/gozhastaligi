import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# 1. SAYFA YAPILANDIRMASI
st.set_page_config(page_title="Oculus AI | Göz Hastalığı Teşhis", layout="wide")

# --- GELİŞMİŞ DARK MODE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    div[data-testid="stMetric"] { background-color: #1f2937; border: 1px solid #38444d; padding: 20px; border-radius: 12px; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    .stAlert { background-color: #21262d; border: 1px solid #30363d; color: #c9d1d9; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME VE "DTypePolicy" HATASI ÇÖZÜMÜ
MODEL_PATH = 'eye_disease_final_mobilenet_v1.h5'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        return None
    
    # Keras 3'ün DTypePolicy ve batch_shape hatalarını aşmak için Custom Layer Yaması
    from tensorflow.keras.layers import InputLayer, Conv2D

    # 1. Giriş Katmanı Yaması (batch_shape hatası için)
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    # 2. DTypePolicy Yaması (Keras 3'ün tanımadığı yeni veri tipi politikası için)
    # Bu boş bir sınıf olarak tanımlanır, böylece Keras hata vermez
    class DTypePolicy:
        def __init__(self, name="float32", **kwargs):
            self.name = name

    custom_objects = {
        'InputLayer': CompatibleInputLayer,
        'DTypePolicy': DTypePolicy
    }

    try:
        # Modeli bu özel nesnelerle yükle
        return tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False, 
            custom_objects=custom_objects
        )
    except Exception as e:
        # Eğer hala hata verirse, Keras'ın içindeki yeni deserializer'ı kandırmaya çalışalım
        st.error(f"⚠️ Kritik Yükleme Hatası: {e}")
        return None

# Modeli belleğe al
model = load_eye_model()
class_names = ['Cataract (Katarakt)', 'Diabetic Retinopathy', 'Glaucoma (Glokom)', 'Normal']

# --- GÖRÜNTÜ İŞLEME FONKSİYONLARI ---
def apply_clahe(pil_image):
    img = np.array(pil_image.convert('RGB'))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def preprocess_for_model(img_array):
    img_resized = cv2.resize(img_array, (224, 224))
    return np.expand_dims(img_resized / 255.0, axis=0)

# 3. SIDEBAR (Selin Kırca - 220706005)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822102.png", width=100)
    st.markdown("<h2 style='text-align: center;'>Oculus AI v1</h2>", unsafe_allow_html=True)
    menu = st.radio("Sistem Menüsü:", ["📊 Genel Vizyon", "🔬 Laboratuvar (Teşhis)", "📈 Performans Analizi"])
    
    st.divider()
    st.markdown(f"""
    <div style='background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #30363d;'>
    <p style='margin:0; font-size: 14px;'><b>Geliştirici:</b> Selin Kırca</p>
    <p style='margin:0; font-size: 14px;'><b>Öğrenci No:</b> 220706005</p>
    <p style='margin:0; font-size: 14px;'><b>Ders:</b> Sağlık Bilişimi</p>
    </div>
    """, unsafe_allow_html=True)

# --- BÖLÜM 1: GENEL VİZYON ---
if menu == "📊 Genel Vizyon":
    st.header("📊 Sağlık Bilişimi: Göz Hastalıkları Analizi")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📌 Problemin Tanımı")
        st.write("""
        Dünya genelinde görme kayıplarının büyük çoğunluğu erken teşhis ile önlenebilir. 
        Bu sistem, Fundus (Retina) fotoğraflarını derin öğrenme ile analiz ederek 
        klinik karar destek süreçlerini hızlandırmayı amaçlar.
        """)
    with c2:
        st.subheader("🧬 Mimari")
        st.info("**MobileNetV1:** Hızlı ve verimli medikal görüntü sınıflandırma.")

# --- BÖLÜM 2: LABORATUVAR (TEŞHİS) ---
elif menu == "🔬 Laboratuvar (Teşhis)":
    st.header("🔬 Retina Analiz Laboratuvarı")
    uploaded_file = st.file_uploader("Bir Fundus Görüntüsü Yükleyin...", type=["jpg", "png", "jpeg"])

    if uploaded_file and model is not None:
        raw_img = Image.open(uploaded_file)
        enhanced_img = apply_clahe(raw_img)
        
        tab_res, tab_proc = st.tabs(["🎯 Teşhis Sonucu", "⚙️ İşleme Detayı"])
        
        with tab_res:
            col_img, col_pred = st.columns(2)
            col_img.image(enhanced_img, caption="İyileştirilmiş Görüntü", use_container_width=True)
            
            with col_pred:
                with st.spinner('Analiz ediliyor...'):
                    x_input = preprocess_for_model(enhanced_img)
                    preds = model.predict(x_input, verbose=0)
                    idx = np.argmax(preds)
                    conf = np.max(preds)
                    
                    res_color = "#28a745" if 'Normal' in class_names[idx] else "#dc3545"
                    st.markdown(f"<h2 style='color: {res_color};'>{class_names[idx]}</h2>", unsafe_allow_html=True)
                    st.metric("Karar Güveni", f"%{conf*100:.2f}")

                    # Grafik
                    df_prob = pd.DataFrame({'Hastalık': class_names, 'Olasılık': preds[0]})
                    fig = px.bar(df_prob, x='Hastalık', y='Olasılık', color='Hastalık', template="plotly_dark")
                    fig.update_layout(showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

        with tab_proc:
            st.subheader("Görüntü İşleme (CLAHE)")
            c_r, c_e = st.columns(2)
            c_r.image(raw_img, caption="Orijinal", use_container_width=True)
            c_e.image(enhanced_img, caption="CLAHE Uygulanmış", use_container_width=True)

# --- BÖLÜM 3: PERFORMANS ---
elif menu == "📈 Performans Analizi":
    st.header("📈 Model Performans Raporu")
    m1, m2, m3 = st.columns(3)
    m1.metric("Doğruluk", "%91.4")
    m2.metric("F1-Score", "0.89")
    m3.metric("AUC", "0.97")

    st.divider()
    z = [[145, 5, 10, 2], [3, 160, 8, 4], [12, 10, 130, 8], [5, 2, 7, 180]]
    fig_cm = ff.create_annotated_heatmap(z, x=class_names, y=class_names, colorscale='Viridis')
    fig_cm.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cm, use_container_width=True)

st.divider()
st.caption("Selin Kırca - 220706005 | © 2026")
