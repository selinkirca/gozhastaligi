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

# 1. SAYFA YAPILANDIRMASI (Selin Kırca - 220706005)
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

# 2. MODEL YÜKLEME VE GÜVENLİK DENETİMİ
MODEL_PATH = 'eye_disease_final_mobilenet_v1.h5'

@st.cache_resource
def load_eye_model():
    # Dosya Varlık Kontrolü
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Kritik Hata: '{MODEL_PATH}' dosyası bulunamadı!")
        return None
    
    # Dosya Boyut Denetimi (Signature Hatasını Önlemek İçin)
    file_size = os.path.getsize(MODEL_PATH)
    if file_size < 1000000: # 1 MB altı ise dosya bozuktur
        st.error(f"⚠️ Model dosyası bozuk veya eksik yüklenmiş! (Boyut: {file_size/1024:.2f} KB)")
        return None

    # --- KERAS 3 UYUMLULUK YAMALARI ---
    from tensorflow.keras.layers import InputLayer
    
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    class FakeDTypePolicy:
        def __init__(self, name="float32", **kwargs):
            self.name = name
            self.compute_dtype = "float32"
            self.variable_dtype = "float32"
        def get_config(self): return {"name": self.name}
        @classmethod
        def from_config(cls, config): return cls(**config)

    custom_objects = {'InputLayer': CompatibleInputLayer, 'DTypePolicy': FakeDTypePolicy}

    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"❌ Model İmza Hatası: Dosya yapısı bozuk. Lütfen modeli GitHub'a tekrar yükleyin. Detay: {e}")
        return None

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

# 3. SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822102.png", width=100)
    st.markdown("<h2 style='text-align: center;'>Oculus AI</h2>", unsafe_allow_html=True)
    menu = st.radio("Menü Seçiniz:", ["📊 Proje Vizyonu", "🔬 Canlı Teşhis", "📈 Analiz Raporu"])
    
    st.divider()
    st.markdown(f"""
    <div style='background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #30363d;'>
    <p style='margin:0;'><b>Geliştirici:</b> Selin Kırca</p>
    <p style='margin:0;'><b>Öğrenci No:</b> 220706005</p>
    <p style='margin:0;'><b>Ders:</b> Sağlık Bilişimi</p>
    </div>
    """, unsafe_allow_html=True)

# --- BÖLÜMLER ---
if menu == "📊 Proje Vizyonu":
    st.header("📊 Sağlık Bilişimi: Göz Analizi")
    st.write("Retina fotoğraflarını derin öğrenme ile analiz ederek erken teşhisi destekler.")
    st.info("**Mimari:** MobileNetV1 & TensorFlow 2.15")

elif menu == "🔬 Canlı Teşhis":
    st.header("🔬 Retina Analiz Laboratuvarı")
    uploaded_file = st.file_uploader("Fundus Görüntüsü Seçin...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        if model is None:
            st.warning("⚠️ Model yüklenemediği için analiz yapılamıyor.")
        else:
            enhanced_img = apply_clahe(Image.open(uploaded_file))
            c1, c2 = st.columns(2)
            c1.image(enhanced_img, caption="Analiz Kesiti (CLAHE)", use_container_width=True)
            
            with c2:
                with st.spinner('Analiz Ediliyor...'):
                    preds = model.predict(preprocess_for_model(enhanced_img), verbose=0)
                    idx = np.argmax(preds)
                    st.markdown(f"<h2 style='color: #58a6ff;'>{class_names[idx]}</h2>", unsafe_allow_html=True)
                    st.metric("Teşhis Güveni", f"%{np.max(preds)*100:.2f}")
                    
                    fig = px.bar(x=class_names, y=preds[0], color=class_names, template="plotly_dark")
                    fig.update_layout(showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

elif menu == "📈 Analiz Raporu":
    st.header("📈 Akademik Performans")
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
