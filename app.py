import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ==========================================================
# 1. Konfigurasi Halaman
# ==========================================================
st.set_page_config(
    page_title="Pakar Cabai AI",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# 2. Injeksi CSS Kustom (Tema Modern Dark-Agri)
# ==========================================================
st.markdown("""
    <style>
    /* Import font Poppins yang lebih modern dan membulat */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

    /* Terapkan font ke seluruh aplikasi */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Menyembunyikan elemen bawaan Streamlit (header, footer, menu) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Styling Judul - Hijau cerah agar kontras di dark mode */
    h1, h2, h3 {
        color: #4ade80 !important; 
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Teks paragraf dan subjudul */
    .st-emotion-cache-10trblm {
        color: #cbd5e1;
        font-weight: 300;
    }

    /* Styling Area Upload (Drag & Drop) */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4ade80;
        border-radius: 16px;
        background-color: rgba(74, 222, 128, 0.05); /* Hijau transparan */
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #22c55e;
        background-color: rgba(74, 222, 128, 0.1);
    }

    /* Mengubah warna teks di dalam kotak upload agar terbaca */
    .st-emotion-cache-9ycgxx {
        color: #f8fafc !important;
    }

    /* Styling Kotak Alert / Info */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
        background-color: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #3b82f6;
        backdrop-filter: blur(10px);
    }
    
    /* Styling Gambar Hasil Output */
    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Styling Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-size: 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #4ade80 !important;
        background-color: rgba(74, 222, 128, 0.1);
    }
    
    /* Garis pemisah */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. Memuat Model
# ==========================================================
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    weights_path = os.path.join(os.path.dirname(__file__), "model.weights.h5")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"File weights tidak ditemukan:\n{weights_path}")

    model.load_weights(weights_path)
    return model

try:
    model = load_model()
    # Notifikasi sukses sengaja dihilangkan agar UI lebih bersih
except Exception as e:
    st.exception(e)
    st.stop()

# ==========================================================
# 4. Data Penyakit & Kelas
# ==========================================================
class_names = ["Curl", "Healthy", "Spot", "Whitefly", "Yellowfish"]

penyakit_info = {
    'Curl': {
        'penyebab': 'Serangan hama serangga penghisap cairan daun (Thrips, Tungau, Aphids).',
        'penanganan': 'Cabut tanaman yang terinfeksi parah. Jaga sanitasi kebun dan aplikasikan insektisida berbahan aktif Abamektin.'
    },
    'Spot': {
        'penyebab': 'Infeksi patogen jamur Cercospora capsici, dipicu oleh kelembaban tinggi.',
        'penanganan': 'Petik daun terinfeksi. Atur jarak tanam untuk sirkulasi udara, semprotkan fungisida berbahan aktif Mankozeb.'
    },
    'Whitefly': {
        'penyebab': 'Hama Kutu Kebul (Bemisia tabaci) yang berkoloni di bagian bawah daun.',
        'penanganan': 'Pasang perangkap lekat kuning (Yellow Sticky Trap) dan semprotkan insektisida nabati berkala.'
    },
    'Yellowfish': {
        'penyebab': 'Infeksi Virus Kuning (Gemini) yang ditularkan melalui gigitan Kutu Kebul.',
        'penanganan': 'Tanaman harus dicabut dan dibakar. Fokus pada pemberantasan vektor kutu kebul.'
    }
}

# ==========================================================
# 5. Header Website
# ==========================================================
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🌶️ Pakar AI: Daun Cabai</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.1em; color: #94a3b8; margin-bottom: 40px;'>
Sistem Deteksi Dini Penyakit Tanaman Cabai Berbasis Deep Learning<br>
Unggah foto daun dan biarkan AI mendiagnosis kondisinya.
</div>
""", unsafe_allow_html=True)

# ==========================================================
# 6. Navigasi & Layout Utama
# ==========================================================
tab1, tab2 = st.tabs(["🔍 Deteksi Cerdas", "📖 Buku Saku Penyakit"])

with tab1:
    col1, col_space, col2 = st.columns([1, 0.1, 1])
    
    with col1:
        st.markdown("### 📷 Modul Input")
        st.info("💡 **Tips:** Pastikan foto fokus pada satu helai daun dengan pencahayaan terang. Jika menggunakan HP, ketuk area di bawah dan pilih 'Kamera'.")
        
        final_image = st.file_uploader("Unggah atau potret daun cabai", type=["jpg", "jpeg", "png"])
        
    with col2:
        st.markdown("### 🔬 Panel Analisis")
        
        if final_image is not None:
            image = Image.open(final_image)
            st.image(image, caption="Citra Input (Telah diproses)", use_container_width=True)
            
            if model is not None:
                with st.spinner("🧠 AI sedang menganalisis pola daun..."):
                    # Preprocessing
                    image_resized = image.resize((128, 128))
                    img_array = np.array(image_resized)
                    
                    if img_array.shape[-1] == 4:
                        img_array = img_array[..., :3]
                        
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Prediksi
                    predictions = model.predict(img_array)
                    class_index = np.argmax(predictions[0])
                    predicted_class = class_names[class_index]
                    confidence = predictions[0][class_index] * 100
                    
                    st.markdown("---")
                    
                    # Filter Penolakan Gambar
                    if confidence < 70.0:
                        st.warning("⚠️ **Akurasi Rendah:** AI tidak yakin ini adalah daun cabai atau gambar terlalu buram. Mohon ambil ulang foto.")
                    else:
                        if predicted_class == "Healthy":
                            st.success(f"🌱 **Status: DAUN SEHAT** (Akurasi: {confidence:.1f}%)")
                            st.info("Kondisi tanaman optimal. Lanjutkan perawatan rutin dan pemberian nutrisi berimbang.")
                        else:
                            st.error(f"🦠 **Terdeteksi: {predicted_class.upper()}** (Akurasi: {confidence:.1f}%)")
                            
                            info = penyakit_info[predicted_class]
                            st.warning(f"**🔍 Penyebab:**\n{info['penyebab']}")
                            st.info(f"**🛠️ Penanganan:**\n{info['penanganan']}")
        else:
            st.markdown("""
            <div style='text-align: center; padding: 50px; background-color: rgba(255,255,255,0.02); border-radius: 12px;'>
                <h4 style='color: #64748b;'>Menunggu Input Gambar...</h4>
                <p style='color: #475569;'>Hasil analisis AI akan muncul di panel ini.</p>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================
# TAB 2: BUKU SAKU
# ==========================================================
with tab2:
    st.markdown("### 📚 Ensiklopedia Penyakit Cabai")
    st.write("Pelajari karakteristik visual dan langkah mitigasi untuk berbagai ancaman pada tanaman cabai Anda.")
    
    # Disederhanakan menggunakan perulangan agar kode lebih rapi
    penyakit_list = ["Curl", "Spot", "Whitefly", "Yellowfish"]
    
    for peny in penyakit_list:
        with st.expander(f"📌 Penyakit: {peny}"):
            try:
                st.image(f"{peny}.jpg", caption=f"Contoh Visual {peny}", width=400)
            except FileNotFoundError:
                st.write(f"*(Gambar {peny}.jpg belum tersedia di direktori)*")
                
            st.markdown(f"**🔬 Penyebab:** {penyakit_info[peny]['penyebab']}")
            st.markdown(f"**🛠️ Tindakan:** {penyakit_info[peny]['penanganan']}")

    with st.expander("🌱 Tanaman Sehat (Healthy)"):
        try:
            st.image("Healthy.jpg", caption="Contoh Daun Sehat", width=400)
        except FileNotFoundError:
            st.write("*(Gambar Healthy.jpg belum tersedia di direktori)*")
        st.markdown("**Ciri-ciri:** Klorofil merata, daun tidak menggulung, dan bebas dari lesi/bercak.")

# ==========================================================
# 7. Footer
# ==========================================================
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.85em; color: #64748b;'>© 2026 | Sistem Klasifikasi Penyakit Daun Cabai | Dibangun dengan Streamlit & TensorFlow</div>", unsafe_allow_html=True)