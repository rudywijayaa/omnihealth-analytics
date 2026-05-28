import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# ---------- Konfigurasi Halaman (Premium Theme) ----------
st.set_page_config(
    page_title='OmniHealth | Premium Analytics',
    page_icon=':shield:',
    layout='centered',
)

# ---------- DETEKSI DIRECTORY AMAN ----------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- ENGINE CUSTOM CSS (Suntikan Desain UI Keren) ----------
st.markdown("""
    <style>
    /* ================= FORCE LIGHT THEME (Mengunci Tampilan Bersih) ================= */
    /* Memaksa background aplikasi tetap putih/terang meskipun device user menggunakan dark mode */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    
    /* FIX RADIKAL: Menghilangkan paksa background putih pada teks label / judul input */
    label, 
    .stWidgetLabel, 
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] p,
    .stSelectbox label,
    .stNumberInput label {
        background-color: transparent !important;
        background: transparent !important;
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* Memaksa Latar Belakang SEMUA Field Input (Number & Selectbox) Menjadi Putih Bersih */
    div[data-baseweb="input"], 
    div[data-baseweb="select"], 
    .stSelectbox div,
    input[type="number"] {
        background-color: #FFFFFF !important;
    }
    
    /* Memaksa teks di dalam input box berwarna hitam agar kontras */
    input, role[textbox], div[data-baseweb="select"] {
        color: #0F172A !important;
    }
    
    /* Mengubah warna tombol plus minus pada number input agar tetap kontras dan bersih */
    button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: none !important;
    }
    
    /* ================= ELIMINASI HEADER GITHUB & MENU BAWAAN ================= */
    /* Menyembunyikan tombol Fork, GitHub, dan status bar hitam di paling atas */
    header, [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Menyembunyikan tombol menu tiga titik bawaan Streamlit di pojok kanan bawah */
    #MainMenu, footer {
        visibility: hidden !important;
    }
    /* ========================================================================= */

    /* Desain Judul Utama (Dibuat super rapat dengan logo) */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #004BAC !important;
        margin: 0;
        padding: 0;
        line-height: 1;
        display: inline-block;
        white-space: nowrap;
    }
    
    /* Tweak Responsif untuk Layar HP */
    @media (max-width: 480px) {
        .main-title {
            font-size: 24px;
        }
    }
    
    .subtitle {
        font-size: 16px;
        color: #64748B !important;
        margin-top: 8px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    /* FIX SIMPEL: GANTI ORANYE JADI BIRU SAAT FOKUS */
    div[data-baseweb="input"]:focus-within, 
    div[data-baseweb="select"]:focus-within {
        border-color: #1E3A8A !important;
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.25) !important;
    }
    
    /* Tweak Kustom Tombol Utama Biru Navy Premium */
    div.stButton > button:first-child {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }
    
    /* Container Premium untuk Hasil Prediksi */
    .result-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.2);
        color: #FFFFFF !important;
        text-align: center;
        margin-top: 25px;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .result-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #C7D2E3 !important;
        margin-bottom: 5px;
    }
    
    .result-value {
        font-size: 42px;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Fungsi Helper Base64 (Mengatasi Isu Gambar Lokal) ----------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

# ---------- Muat Model & Scaler (Cached & Safe Path) ----------
@st.cache_resource
def load_artefak():
    model_path  = os.path.join(CURRENT_DIR, 'regresi_berganda.pkl')
    scaler_path = os.path.join(CURRENT_DIR, 'scaler.pkl')
    fitur_path  = os.path.join(CURRENT_DIR, 'fitur.pkl')
    
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    fitur  = joblib.load(fitur_path)
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()


# ---------- Header Aplikasi (Logo Path Diperbaiki Secara Absolut) ----------
logo_path = os.path.join(CURRENT_DIR, "static", "logo.png")
img_base64 = get_base64_image(logo_path)

if img_base64:
    st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-top: 15px; margin-bottom: 5px;">
            <img src="data:image/png;base64,{img_base64}" width="55" style="object-fit: contain; display: inline-block; margin: 0; padding: 0;">
            <span class="main-title">OmniHealth Analytics</span>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<h1 style="text-align: center; color: #004BAC; font-weight: 800;">OmniHealth Analytics</h1>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Medical Insurance Premium Predictive Dashboard</div>', unsafe_allow_html=True)


# ---------- Area Input Form ----------
st.markdown("""
    <div style="border-top: 1px solid #E2E8F0; margin-top: 30px; padding-top: 20px; margin-bottom: 15px;">
        <p style="font-weight: 800; font-size: 20px; color: #0F172A; margin: 0; letter-spacing: -0.3px;">
            📋 Profil & Demografi Calon Nasabah
        </p>
    </div>
""", unsafe_allow_html=True)

# Grid Kolom Atas (Numerik)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input('Usia (Tahun)', min_value=18, max_value=100, value=25, step=1)
with col2:
    bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=10.0, max_value=60.0, value=24.5, step=0.1, format='%.1f')
with col3:
    children = st.number_input('Jumlah Tanggungan Anak', min_value=0, max_value=10, value=0, step=1)

# Grid Kolom Bawah (Kategorikal)
col4, col5 = st.columns(2)
with col4:
    sex = st.selectbox('Jenis Kelamin', ['Perempuan', 'Laki-laki'])
with col5:
    smoker = st.selectbox('Status Merokok', ['Tidak Merokok', 'Perokok Aktif'])


# ---------- Logika Pemetaan Fitur ke Model ----------
sex_male = 1 if sex == 'Laki-laki' else 0
smoker_yes = 1 if smoker == 'Perokok Aktif' else 0

input_mentah = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_male': sex_male,
    'smoker_yes': smoker_yes,
    'region_northwest': 0,
    'region_southeast': 0,
    'region_southwest': 0
}


# ---------- Tombol Hitung Eksekusi ----------
hitung = st.button('Cek Estimasi Premi Sekarang', type='primary', use_container_width=True)


# ---------- Area Hasil Eksekusi & Parameter Transparansi ----------
if hitung:
    try:
        nilai = pd.DataFrame([[input_mentah[f] for f in FITUR]], columns=FITUR)
        nilai_sc = scaler.transform(nilai)
        
        pred_log = model.predict(nilai_sc)[0]
        pred_asli = np.expm1(pred_log)

        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Estimasi Premi Tahunan Yang Direkomendasikan</div>
                <div class="result-value">${pred_asli:,.2f}</div>
                <p style="font-size: 12px; color: #94A3B8; margin-top: 10px; margin-bottom: 0;">
                    *Dihitung secara otomatis menggunakan sistem Predictive Modeling teroptimasi skala logaritma.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔮 Lihat Cara Model AI Menghitung Premi Anda"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Skor Prediksi Dasar (Log)", f"{pred_log:.4f}")
            with col_b:
                st.metric("Nilai Dasar Premi (Intercept)", f"{model.intercept_:.4f}")
            
            kamus_fitur = {
                'age': 'Usia',
                'bmi': 'Indeks Massa Tubuh (BMI)',
                'children': 'Jumlah Anak / Tanggungan',
                'sex_male': 'Jenis Kelamin',
                'smoker_yes': 'Status Perokok'
            }

            df_koef = pd.DataFrame({
                'Faktor Risiko': FITUR,
                'Bobot Pengaruh (β)': model.coef_.round(4)
            })
            
            df_koef = df_koef[df_koef['Faktor Risiko'].isin(kamus_fitur.keys())].copy()
            df_koef['Faktor Risiko'] = df_koef['Faktor Risiko'].map(kamus_fitur)

            st.dataframe(df_koef, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f'Terjadi kendala teknis pada core model: {e}')
else:
    st.markdown('<p style="text-align: center; font-size: 13px; color: #64748B; margin-top: 15px;">💡 Masukkan profil nasabah di atas lalu klik tombol untuk menguji kecerdasan model.</p>', unsafe_allow_html=True)


# ---------- Area Footer Korporat ----------
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown('<p style="text-align: center; font-size: 12px; color: #94A3B8;">© 2026 OmniHealth Analytics Corp. | Rudy Wijaya | PPKD Jakarta Selatan — Kejuruan Data Analyst</p>', unsafe_allow_html=True)