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
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment
import numpy as np
import tempfile
import torch
import librosa
import pandas as pd # <--- Necesario para la gráfica de barras

st.set_page_config(page_title="LALAAI", layout="wide")

# =========================================================
# === 1. CARGA DINÁMICA DE MÓDULOS DE PROCESAMIENTO (AI) ===
# =========================================================

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

# Carga de Módulos
try:
    # Carga de la función de Energía
    energy_module = load_module_dynamically("energy_module", '../Energy/energy.py')
    calculate_track_energy = energy_module.calculate_track_energy
    
    # Se añade la carga de Métricas
    metrics_module = load_module_dynamically("metrics_module", '../metricas/metricas.py')
    calculate_producer_metrics = metrics_module.calculate_producer_metrics # <--- Nueva función
    
    predict_module = load_module_dynamically("ai_model_predict", '../AI-Model/predict.py')
    predict_song = predict_module.predict_song

except Exception as e:
    st.error(f"Error al cargar módulos de AI/Métricas: {e}")
    # Define funciones dummy para evitar que el código falle si hay error
    def calculate_track_energy(path): return np.linspace(0, 10, 400), np.sin(np.linspace(0, 10, 400)) * np.exp(-np.linspace(0, 10, 400)/5)
    def predict_song(path): return "Deep House", {"Deep House": 0.6, "Progressive House": 0.3, "Ambient": 0.1}
    # Nuevo dummy para la métrica
    def calculate_producer_metrics(path): return {"Energía RMS": 45, "Brillo (Centroide)": 55, "Energía de Graves (<80Hz)": 30, "Rango Dinámico (DR)": 70}


# =========================================================
# === 2. FUNCIÓN DE SEPARACIÓN (DEMUCS) ===
# =========================================================

@st.cache_resource
def get_demucs_model(model_name="htdemucs"):
    """Carga el modelo Demucs una sola vez y lo guarda en caché."""
    return get_model(model_name).to("cpu").eval()

def load_audio_pydub(path, target_sr=44100):
    """Carga cualquier audio (MP3/WAV) usando pydub y devuelve np.array normalizado y sample rate."""
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples, target_sr

def save_as_mp3(audio_array, sr):
    """Guarda un numpy array de audio como MP3 usando pydub en un archivo temporal."""
    audio_int16 = (audio_array * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    tmp_path = tempfile.mktemp(suffix=".mp3")
    audio_segment.export(tmp_path, format="mp3")
    return tmp_path

def separate_audio_stems(input_path):
    """Separa el audio en stems (vocals y accompaniment) usando Demucs y pydub."""
    
    # 1. Cargar audio
    y, sr = load_audio_pydub(input_path, target_sr=44100)
    wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)  # [1, canales, samples]

    # 2. Separación con Demucs
    model = get_demucs_model()
    sources = apply_model(model, wav, device="cpu")[0]  # [stems, channels, samples]

    # 3. Crear stems
    vocals = sources[3].mean(axis=0).numpy()
    accompaniment = (sources[0] + sources[1] + sources[2]).mean(axis=0).numpy()

    # 4. Guardar stems como MP3
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
            st.session_state.audio_path = tmp_file.name 
            audio_path = tmp_file.name
            
        pred_genre, probs_dict = predict_song(audio_path)
        
        if not isinstance(probs_dict, dict):
            st.error(f"Error Crítico de Datos: Se esperaba un diccionario, se recibió {type(probs_dict)}.")
            st.stop() 
        
        times, rms = calculate_track_energy(audio_path)
        
        producer_metrics = calculate_producer_metrics(audio_path)
        
        st.header(f"  Genre Detected: **{pred_genre}**")
        
        # 3. GRÁFICA DE DONUT Y GRÁFICA DE BARRAS
        col_genre, col_spacer, col_metrics = st.columns([1.5, 0.1, 1]) 
        
        with col_genre:
            st.markdown("##### Probabilidades")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie(list(probs_dict.values()), labels=list(probs_dict.keys()),
                    autopct='%1.1f%%', startangle=90,
                    colors=WARM_PALETTE[:len(probs_dict)])
            ax1.set_title("") 
            st.pyplot(fig1, use_container_width=True)
        
        # Gráfico de Barras de Métricas Clave
        with col_metrics:
            st.markdown("##### KEY METRICS")
            
            # Convertir el diccionario a un DataFrame con pandas
            df_metrics = pd.DataFrame(
                list(producer_metrics.items()), 
                columns=['Métrica', 'Valor']
            )

            # Crear el gráfico de barras con plotly
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=df_metrics['Métrica'],
                        y=df_metrics['Valor'],
                        marker_color=WARM_PALETTE[:len(df_metrics)], # Asegura que los colores coincidan con el número de métricas
                        text=[f"{v:.1f}" for v in df_metrics['Valor']],
                        textposition='auto',
                        width=0.9 
                    )
                ]
            )

            # Estilos del gráfico
            fig_bar.update_layout(
                # 🚨 CAMBIO 1: Aumentar la altura para que ocupe más espacio
                height=300, # Ajusta esta altura según lo necesites para que baje más
                margin=dict(l=10, r=10, t=20, b=10),
                # 🚨 CAMBIO 2: Fondos en blanco
                plot_bgcolor="white", 
                paper_bgcolor="white",
                yaxis=dict(range=[0, 100], title="Relative Punctuation(0-100)"),
                # 🚨 CAMBIO 3: Eliminar el título del eje X
                xaxis_title=None
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 4. GRÁFICA DE ENERGÍA (RMS)
        # La barra "---" se ha eliminado en el commit anterior
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=rms, mode="lines", line=dict(color="#ff9900", width=2)))
        mean_rms = np.mean(rms)
        fig.add_hline(y=mean_rms, line_dash="dash", line_color="#33AFFF", 
                      annotation_text=f"Average Energy: {mean_rms:.3f}", 
                      annotation_position="bottom right", annotation_font_size=10)
        fig.update_layout(
            # 🚨 CAMBIO 4: Ajustar la altura para que el gráfico sea más alto
            height=380, # Ajusta este valor si necesitas que baje aún más
            margin=dict(l=10, r=10, t=10, b=10), 
            # 🚨 CAMBIO 5: Fondos en blanco
            paper_bgcolor="white",
            plot_bgcolor="white", 
            title=dict(text="Energy Analysis over Time (t)", font=dict(size=14, color="#333")),
            xaxis_title="Time (s)", yaxis_title="Amplitud RMS", showlegend=False
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
