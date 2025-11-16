import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sys
import os
import importlib.util
# Importaciones necesarias para Demucs (Stem Splitter)
import torch
import librosa
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment
import numpy as np
import tempfile
import torch
import librosa

st.set_page_config(page_title="LALAAI", layout="wide")

# =========================================================
# === 1. CARGA DINÁMICA DE MÓDULOS DE PROCESAMIENTO (AI) ===
# ============================================s=============

# ... (El código de load_module_dynamically y la carga de predict_song, etc., se mantiene igual) ...

# Función auxiliar para cargar módulos dinámicamente
def load_module_dynamically(module_name, relative_path):
    full_path = os.path.normpath(os.path.join(os.path.dirname(__file__), relative_path))
    module_dir = os.path.dirname(full_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Carga de Módulos (usando st.cache_resource para modelos grandes si fuera necesario, pero aquí simple)
try:
    energy_module = load_module_dynamically("energy_module", '../Energy/energy.py')
    calculate_track_energy = energy_module.calculate_track_energy
    
    spotify_module = load_module_dynamically("spotify_module", '../Spotify/Spotify.py')
    get_recommended_tracks = spotify_module.get_recommended_tracks
    
    predict_module = load_module_dynamically("ai_model_predict", '../AI-Model/predict.py')
    predict_song = predict_module.predict_song

except Exception as e:
    st.error(f"Error al cargar módulos de AI/Spotify: {e}")
    # Define funciones dummy para evitar que el código falle si hay error
    def calculate_track_energy(path): return np.linspace(0, 10, 400), np.sin(np.linspace(0, 10, 400)) * np.exp(-np.linspace(0, 10, 400)/5)
    def predict_song(path): return "Deep House", {"Deep House": 0.6, "Progressive House": 0.3, "Ambient": 0.1}
    def get_recommended_tracks(probs, total): return [{"name": f"Track {i}", "artists": [{"name": "Artista Falso"}], "album": {"images": [{"url": None}]}} for i in range(total)]


# =========================================================
# === 2. FUNCIÓN DE SEPARACIÓN (DEMUCS) ===
# =========================================================

@st.cache_resource
def get_demucs_model(model_name="htdemucs"):
    """Carga el modelo Demucs una sola vez y lo guarda en caché."""
    return get_model(model_name).to("cpu").eval()

def separate_audio_stems(input_path):
    """Carga, separa y guarda los stems de voces y acompañamiento sin usar soundfile."""
    
    # 1. Cargar audio
    y, sr = librosa.load(input_path, sr=44100, mono=False)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)  # [canales, samples]

    wav = torch.from_numpy(y).float().unsqueeze(0)  # [1, canales, samples]

    # 2. Separación con Demucs (modelo en caché)
    model = get_demucs_model()
    sources = apply_model(model, wav, device="cpu")[0]  # [stems, channels, samples]

    vocals = sources[3].mean(axis=0).numpy()
    accompaniment = (sources[0] + sources[1] + sources[2]).mean(axis=0).numpy()

    # 3. Guardar archivos temporales como MP3 usando pydub
    def save_as_mp3(audio_array, sr):
        # Normalizar a int16
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,  # 16-bit
            channels=1 if audio_array.ndim == 1 else audio_array.shape[0]
        )
        tmp_path = tempfile.mktemp(suffix=".mp3")
        audio_segment.export(tmp_path, format="mp3")
        return tmp_path

    vocals_path = save_as_mp3(vocals, sr)
    music_path = save_as_mp3(accompaniment, sr)

    return vocals_path, music_path, sr

# =========================================================
# ==================== 3. CUSTOM CSS ======================
# =========================================================

st.markdown("""
<style>
     /* ---------------------- ANULACIÓN DE COLOR PRIMARIO DE STREAMLIT ---------------------- */   
     /* Definición de variables primarias (esto debería funcionar si Streamlit las respeta) */
     :root {
         --primary-color: #ffd700;
         --primary-text-color: #000000;
         --primary-background-color: #ffb300; 
     }  
     /* **CORRECCIÓN:** Selector ultra-específico para el botón st.button(type="primary") */
     /* Apuntamos al contenedor específico con la clase 'primary' que Streamlit aplica */
     div.stButton > button[data-testid*="stButton"] {
         background-color: var(--primary-color) !important;
         color: var(--primary-text-color) !important;
         border-color: var(--primary-background-color) !important;
         font-weight: bold;
     }  
     div.stButton > button[data-testid*="stButton"]:hover {
         background-color: var(--primary-background-color) !important;
         border-color: #ff9900 !important;
     }
    
     /* Contenedor principal del st.info (ya estaba bien) */
     div[data-testid="stAlert"] [data-baseweb="button"] {
         background-color: #fff7e6 !important; /* Fondo amarillo claro */
         color: #333333 !important; /* Texto gris oscuro */
         border-left-color: #ffb300 !important; /* Barra lateral amarilla */
     }
    
     /* Icono del st.info (ya estaba bien) */
     div[data-testid="stAlert"] [data-baseweb="button"] svg {
         fill: #ffb300 !important; 
     }  
     /* [RESTO DE TUS ESTILOS DE CARD, UPLOAD, NAVBAR Y LIMPIEZA...] */
    
     /* CARD STYLE ADDED FOR CLEAN LOOK */
     .card {
         background-color: #f7f7f7;
         border-radius: 12px;
         padding: 20px;
         margin-bottom: 20px;
         box-shadow: 0 4px 10px rgba(0,0,0,0.05);
     }
    
     /* UPLOAD BOX FOR EMPTY STATE */
     .upload-box {
         height: 280px;
         border: 3px dashed #ffb300;
         border-radius: 10px;
         display: flex;
         justify-content: center;
         align-items: center;
         font-size: 18px;
         color: #888;
         background-color: #fff7e6;
         margin-top: 10px;
         margin-bottom: 20px;
     }
    
     /* NAVBAR */
     .top-bar {
         position: fixed;
         top: 0;
         left: 0;
         width: 100%;
         z-index: 9999;
         background: linear-gradient(90deg, #ffd700, #ffb300);
         padding: 18px 40px;
         font-size: 22px;
         font-weight: 600;
         color: #000;
         border-radius: 0 0 15px 15px;
         box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
     }  
     .centered-content {
         text-align: center;
         display: flex;
         flex-direction: column;
         align-items: center; 
     }
    
     /* ---------------------- LIMPIEZA Y AJUSTES ---------------------- */
     /* REMOVE STREAMLIT DEFAULT PADDING & WHITE BLOCKS (CORREGIDO) */
     .main > div {
         padding-top: 0 !important;
         margin-top: 0 !important;
         background: transparent !important;
         box-shadow: none !important;
     }
     /* REMOVE STREAMLIT DEFAULT HEADER */
     header, .st-emotion-cache-18ni7ap {
         display: none !important;
         visibility: hidden !important;
         height: 0 !important;
     }
    
     /* FIX SIDEBAR SHIFT */
     section[data-testid="stSidebar"] {
         padding-top: 0 !important;
         margin-top: 0 !important;
     }
     /* FINAL FIX: SET EXACT SPACE BETWEEN NAVBAR AND FIRST ELEMENT */
     .block-container {
         padding-top: 0 !important; /* Asegura que el contenedor no tenga padding superior */
         margin-top: 1px !important; 
     }

</style>
""", unsafe_allow_html=True)

# ===================== 4. INTERFAZ =======================
WARM_PALETTE = ['#FFD700', '#FFB300', '#FF8C00', '#D2691E', '#8B4513']
# ------------ NAVBAR ------------
st.markdown("<div class='top-bar'>Prod.AI — Music Genre Detector & Stem Splitter </div>", unsafe_allow_html=True)

# ------------ LAYOUT -------------
left, right = st.columns([2, 1], gap="large")
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'stem_results' not in st.session_state:
    st.session_state.stem_results = None

# ------------ LEFT PANEL (Carga y Análisis) -------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Drop your audio file")
    audio = st.file_uploader(" ", type=["mp3","wav","m4a"], label_visibility="collapsed")

    if audio:
        # Guardar archivo temporalmente y actualizar estado
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.type.split('/')[-1]}") as tmp_file:
            tmp_file.write(audio.read())
            st.session_state.audio_path = tmp_file.name # Guardamos la ruta en el estado
            audio_path = tmp_file.name
        
        # El resto de la lógica de análisis (Predicción y Gráficas) se mantiene igual
        pred_genre, probs_dict = predict_song(audio_path)
        tracks = get_recommended_tracks(probs_dict, total_tracks=3)
        times, rms = calculate_track_energy(audio_path)
        
        st.header(f"   Genre Detected: **{pred_genre}**")
        
        # 3. GRÁFICA DE DONUT Y RECOMENDACIONES (PRIMERO)
        col_genre, col_spacer, col_spotify = st.columns([1.5, 0.1, 1]) 
        
        with col_genre:
            st.markdown("##### Probabilidades")
            fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
            ax1.pie(list(probs_dict.values()), labels=list(probs_dict.keys()),
                    autopct='%1.1f%%', startangle=90,
                    colors=WARM_PALETTE[:len(probs_dict)])
            ax1.set_title("") 
            st.pyplot(fig1, use_container_width=True)
        
        with col_spotify:
            st.markdown("##### Recomendaciones de Spotify")
            for track in tracks:
                track_name = track.get("name", "Desconocido")
                artist_name = track.get("artists", [{"name": "Desconocido"}])[0]["name"]
                st.markdown(f"**{track_name}**")
                st.caption(f"🎧 {artist_name}")
                st.markdown("---")
        
        # 4. GRÁFICA DE ENERGÍA 
        st.markdown("---") 
        # (Lógica de la gráfica de energía de Plotly se mantiene)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=rms, mode="lines", line=dict(color="#ff9900", width=2)))
        mean_rms = np.mean(rms)
        fig.add_hline(y=mean_rms, line_dash="dash", line_color="#33AFFF", 
                      annotation_text=f"Energía promedio: {mean_rms:.3f}", 
                      annotation_position="bottom right", annotation_font_size=10)
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#fff7e6",
            plot_bgcolor="#fff7e6", title=dict(text="Análisis de Energía (RMS)", font=dict(size=14, color="#333")),
            xaxis_title="Tiempo (s)", yaxis_title="Amplitud RMS", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.session_state.audio_path = None # Limpiar ruta si no hay audio
        st.markdown("<div class='upload-box'>Drag & drop an audio file here</div>",
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------ RIGHT PANEL (Stem Splitter) -------------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎛 Stem Splitter")
    
    if st.session_state.audio_path is None:
        st.info("Upload an audio file to enable stem separation.")
    else:
        st.write("Click to separate Stems (Acapella & Instrumental).")
        
        # Botón de separación
        if st.button(" Stem Separation (Acapella & Instrumental)", use_container_width=True, type="primary"):
            
            with st.spinner("Separating stems... This may take a few minutes."):
                vocals_path, music_path, sr = separate_audio_stems(st.session_state.audio_path)
                st.session_state.stem_results = (vocals_path, music_path)
            
            st.success("Separation Completed!.")
        
        # Mostrar botones de descarga si la separación se ha realizado
        if st.session_state.stem_results:
            vocals_path, music_path = st.session_state.stem_results
            
            # Reutilizamos el título de la tarjeta del panel derecho
            st.markdown("---")
            st.markdown("##### Downloads")

            # Botones de descarga
            st.download_button(
                label="⬇ Download Vocals (Acapella)",
                data=open(vocals_path, "rb").read(),
                file_name="vocals.mp3",
                mime="audio/mp3",
                use_container_width=True
            )
            st.download_button(
                label="Download music (Instrumental)",
                data=open(music_path, "rb").read(),
                file_name="instrumental.mp3",
                mime="audio/mp3",
                use_container_width=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

