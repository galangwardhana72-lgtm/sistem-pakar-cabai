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
    layout="wide",
    initial_sidebar_state="collapsed"
)
# CSS kustom untuk memperbesar tampilan kamera di mobile
st.markdown("""
    <style>
    [data-testid="stCameraInput"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    [data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
    }
    </style>
""", unsafe_allow_html=True)
# ==========================================================
# 2. Memuat Model (Arsitektur + Weights)
# ==========================================================
@st.cache_resource
def load_model():

    # Membuat arsitektur CNN (HARUS sama seperti saat training)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            5,
            activation="softmax"
        )
    ])

    # Lokasi file weights
    weights_path = os.path.join(
        os.path.dirname(__file__),
        "model.weights.h5"
    )

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"File weights tidak ditemukan:\n{weights_path}"
        )

    # Memuat bobot model
    model.load_weights(weights_path)

    return model


try:
    model = load_model()
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.exception(e)
    st.stop()

# ==========================================================
# Nama Kelas
# ==========================================================
class_names = [
    "Curl",
    "Healthy",
    "Spot",
    "Whitefly",
    "Yellowfish"
]
# Kamus Data Penyakit untuk ditampilkan langsung di layar analisis
penyakit_info = {
    'Curl': {
        'penyebab': 'Serangan hama serangga penghisap cairan daun seperti Thrips, Tungau, atau Aphids.',
        'penanganan': 'Cabut dan musnahkan tanaman yang sudah terinfeksi parah. Jaga sanitasi kebun dari gulma, dan aplikasikan insektisida berbahan aktif Abamektin.'
    },
    'Spot': {
        'penyebab': 'Infeksi patogen jamur Cercospora capsici, biasanya dipicu oleh kondisi kebun yang terlalu lembab.',
        'penanganan': 'Segera petik dan buang daun yang terinfeksi. Atur jarak tanam agar sirkulasi udara lancar, dan semprotkan fungisida berbahan aktif Mankozeb.'
    },
    'Whitefly': {
        'penyebab': 'Hama serangga kecil Kutu Kebul (Bemisia tabaci) yang berkoloni di bagian bawah daun.',
        'penanganan': 'Pasang perangkap lekat kuning (Yellow Sticky Trap) di area kebun dan lakukan penyemprotan insektisida nabati secara berkala.'
    },
    'Yellowfish': {
        'penyebab': 'Infeksi Virus Kuning (Gemini) yang ditularkan melalui gigitan hama Kutu Kebul.',
        'penanganan': 'Tidak ada obat kuratif. Tanaman harus segera dicabut dan dibakar agar tidak menular. Pengendalian difokuskan pada pemberantasan vektor kutu kebul.'
    }
}

# 3. Header Website
st.markdown("<h1 style='text-align: center;'>Sistem Pakar AI: Deteksi Penyakit Daun Cabai</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.1em; margin-bottom: 30px;'>
Aplikasi cerdas berbasis Convolutional Neural Network (CNN) untuk mendeteksi dini penyakit pada tanaman cabai. <br>
Kenali penyakitnya, pahami penanganannya, dan selamatkan panen Anda.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# 4. Membuat Tabs untuk Navigasi UI
tab1, tab2 = st.tabs(["Deteksi Cerdas", "Buku Saku Penyakit"])

# ==========================================
# TAB 1: DETEKSI AI
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Modul Input Gambar")
        st.info("Pastikan foto fokus pada satu helai daun cabai dengan pencahayaan yang merata.")
        
        input_method = st.radio("Pilih metode input:", ("Unggah Dokumen", "Aktivasi Kamera"))
        
        final_image = None
        
        if input_method == "Aktivasi Kamera":
            camera_photo = st.camera_input("Ambil foto langsung")
            if camera_photo:
                final_image = camera_photo
        else:
            uploaded_file = st.file_uploader("Seret dan letakkan file gambar di sini", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                final_image = uploaded_file
        
    with col2:
        st.subheader("Panel Analisis")
        if final_image is not None:
            image = Image.open(final_image)
            st.image(image, caption='Citra Input', use_container_width=True)
            
            if model is not None:
                with st.spinner("Memproses klasifikasi citra..."):
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
                    
                    # Filter Penolakan Gambar Bukan Daun
                    if confidence < 70.0:
                        st.warning("Kepercayaan model rendah. Pastikan objek adalah daun cabai yang fokus.")
                    else:
                        if predicted_class == 'Healthy':
                            st.success(f"Status: Daun Sehat (Tingkat Akurasi: {confidence:.2f}%)")
                            st.info("Kondisi tanaman sangat baik. Lanjutkan aplikasi nutrisi berimbang (NPK) dan pemantauan rutin.")
                        else:
                            st.error(f"Terdeteksi: {predicted_class} (Tingkat Akurasi: {confidence:.2f}%)")
                            
                            # Menampilkan deskripsi penyebab dan penanganan secara dinamis
                            info = penyakit_info[predicted_class]
                            st.warning(f"**Disebabkan oleh:** {info['penyebab']}")
                            st.info(f"**Cara Penanganan:** {info['penanganan']}")
        else:
            st.write("Menunggu input gambar untuk memulai analisis...")

# ==========================================
# TAB 2: INFORMASI PENYAKIT & PENANGANAN
# ==========================================
with tab2:
    st.header("Buku Saku Pengendalian Hama Terpadu")
    st.write("Prosedur penanganan detail berdasarkan pedoman agrikultur standar:")
    
    with st.expander("Daun Keriting (Curl)"):
        try:
            st.image("Curl.jpg", caption="Contoh Daun Keriting (Curl)", width=400)
        except FileNotFoundError:
            st.info("Gambar 'Curl.jpg' belum tersedia di folder.")
            
        st.write(f"""
        **Disebabkan oleh:** {penyakit_info['Curl']['penyebab']}
        
        **Gejala:** Helai daun melengkung ke atas atau ke bawah, menebal, mengerut, dan pertumbuhan vegetatif terhambat.
        
        **Cara Penanganan:** {penyakit_info['Curl']['penanganan']}
        """)

    with st.expander("Bercak Daun (Spot / Cercospora)"):
        try:
            st.image("Spot.jpg", caption="Contoh Bercak Daun (Spot)", width=400)
        except FileNotFoundError:
            st.info("Gambar 'Spot.jpg' belum tersedia di folder.")
            
        st.write(f"""
        **Disebabkan oleh:** {penyakit_info['Spot']['penyebab']}
        
        **Gejala:** Terdapat lesi sirkular dengan pusat abu-abu dan tepi coklat tua. Daun menguning klorosis dan berpotensi absisi (gugur).
        
        **Cara Penanganan:** {penyakit_info['Spot']['penanganan']}
        """)

    with st.expander("Kutu Kebul (Whitefly)"):
        try:
            st.image("Whitefly.jpg", caption="Contoh Kutu Kebul (Whitefly)", width=400)
        except FileNotFoundError:
            st.info("Gambar 'Whitefly.jpg' belum tersedia di folder.")
            
        st.write(f"""
        **Disebabkan oleh:** {penyakit_info['Whitefly']['penyebab']}
        
        **Gejala:** Koloni serangga putih di abaksial (bawah) daun. Ekskresi embun madu memicu pertumbuhan kapang jelaga.
        
        **Cara Penanganan:** {penyakit_info['Whitefly']['penanganan']}
        """)
        
    with st.expander("Daun Kuning (Yellowfish / Virus Gemini)"):
        try:
            st.image("Yellowfish.jpg", caption="Contoh Daun Kuning (Yellowfish)", width=400)
        except FileNotFoundError:
            st.info("Gambar 'Yellowfish.jpg' belum tersedia di folder.")
            
        st.write(f"""
        **Disebabkan oleh:** {penyakit_info['Yellowfish']['penyebab']}
        
        **Gejala:** Penebalan tulang daun dengan klorosis kuning mencolok. Tanaman kerdil dan gagal berproduksi.
        
        **Cara Penanganan:** {penyakit_info['Yellowfish']['penanganan']}
        """)
        
    with st.expander("Daun Sehat (Healthy)"):
        try:
            st.image("Healthy.jpg", caption="Contoh Daun Sehat", width=400)
        except FileNotFoundError:
            st.info("Gambar 'Healthy.jpg' belum tersedia di folder.")
            
        st.write("""
        **Karakteristik Visual:** Klorofil merata, morfologi daun normal, tidak terdapat lesi patogen.
        
        **Cara Pemeliharaan:** Lanjutkan aplikasi nutrisi berimbang (NPK) dan monitoring berkala.
        """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.85em;'>2026 | Sistem Klasifikasi Penyakit Daun Cabai | Dibangun dengan Streamlit & TensorFlow</div>", unsafe_allow_html=True)