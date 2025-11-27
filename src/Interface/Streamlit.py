import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sys
import os
import importlib.util
import torch
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model
from pydub import AudioSegment
import numpy as np
import tempfile
import torch
import librosa
import pandas as pd 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Prod.AI", layout="wide")

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
    calculate_producer_metrics = metrics_module.calculate_producer_metrics 
    
    predict_module = load_module_dynamically("ai_model_predict", '../AI-Model/predict.py')
    predict_song = predict_module.predict_song

except Exception as e:
    # Define funciones dummy para evitar que el código falle si hay error
    def calculate_track_energy(path): return np.linspace(0, 10, 400), np.sin(np.linspace(0, 10, 400)) * np.exp(-np.linspace(0, 10, 400)/5)
    def predict_song(path): return "Deep House", {"Deep House": 0.6, "Progressive House": 0.3, "Ambient": 0.1}
    # CORRECCIÓN: Se usan las claves simplificadas que coinciden con GENRE_TARGETS
    def calculate_producer_metrics(path): return {"RMS": 45, "Centroid": 55, "Lows": 30, "DR": 70}


# =========================================================
# === 2. FUNCIÓN DE SEPARACIÓN (DEMUCS) - CORREGIDA ===
# =========================================================
@st.cache_resource
def get_demucs_model(model_name="htdemucs"):
    """Carga el modelo Demucs una sola vez y lo guarda en caché."""
    return get_model(model_name).to("cpu").eval()

def load_audio_pydub(path, target_sr=44100):
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(2).set_frame_rate(target_sr)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    y = samples.reshape((-1, 2)).T / 32768.0
    y = y.astype(np.float32)
    return y, target_sr

def separate_audio_stems(input_path):
    y, sr = load_audio_pydub(input_path, target_sr=44100)
    wav = torch.from_numpy(y).float().unsqueeze(0)  
    model = get_demucs_model()
    sources = apply_model(model, wav, device="cpu")[0]  
    vocals = sources[3].mean(axis=0).numpy()
    accompaniment = (sources[0] + sources[1] + sources[2]).mean(axis=0).numpy()  
    vocals_path = save_as_mp3(vocals, sr)
    music_path = save_as_mp3(accompaniment, sr)
    return vocals_path, music_path, sr

def save_as_mp3(audio_array, sr):
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

# =========================================================
# === 3. DATOS Y FUNCIÓN PARA GRÁFICO RADAR ===
# =========================================================

# Valores simulados (0-100) para las métricas clave.
GENRE_TARGETS = {
    "Deep House": {
        "RMS": 65,
        "Centroid": 40,
        "Lows": 75,
        "DR": 55
    },
    "Techno": {
        "RMS": 85,
        "Centroid": 70,
        "Lows": 60,
        "DR": 45
    },
    "Progressive House": {
        "RMS": 70,
        "Centroid": 50,
        "Lows": 65,
        "DR": 65
    },
    "Ambient": {
        "RMS": 30,
        "Centroid": 25,
        "Lows": 40,
        "DR": 80
    },
    "Pop": {
        "RMS": 90,
        "Centroid": 80,
        "Lows": 50,
        "DR": 35
    }
}

def create_radar_chart(user_metrics: dict, target_genre: str):
    """
    Crea un gráfico de radar comparando las métricas del usuario con el promedio del género target.
    """
    target_metrics = GENRE_TARGETS.get(target_genre, GENRE_TARGETS["Deep House"]) 
    
    categories = list(target_metrics.keys()) 
    print(categories)
    # Aquí es donde se usa .get(k, 0). Si las claves no coinciden, devuelve 0.
    user_data = [user_metrics.get(k, 0) for k in categories] 
    target_data = list(target_metrics.values())

    df_radar = pd.DataFrame({
        'Métrica': categories * 2,
        'Valor': user_data + target_data,
        'Tipo': ['Tu Canción'] * len(categories) + [f'{target_genre} Target'] * len(categories)
    })

    fig_radar = go.Figure()

    # Trazar Tu Canción
    fig_radar.add_trace(go.Scatterpolar(
        r=user_data,
        theta=categories,
        fill='toself',
        name='Tu Canción',
        marker_color='#FA8F46' 
    ))

    # Trazar el Target (Género)
    fig_radar.add_trace(go.Scatterpolar(
        r=target_data,
        theta=categories,
        fill='toself',
        name=f'{target_genre} Target',
        marker_color='#F08E26',
        opacity=0.5
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False, 
        height=320,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="white",  
        paper_bgcolor="white",
        font_color="#333333"
    )

    return fig_radar


# =========================================================
# ==================== 4. CUSTOM CSS ======================
# =========================================================

# NOTA: Todo el CSS está aquí. No se ha perdido. Solo hay que asegurar que Streamlit lo inyecte.
st.markdown("""
<style>
    /* ---------------------- ANULACIÓN DE COLOR PRIMARIO DE STREAMLIT ---------------------- */  
    :root {
        --primary-color: #ffd700;
        --primary-text-color: #000000;
        --primary-background-color: #ffb300;  
    }  
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
    /* Contenedor principal del st.info */
    div[data-testid="stAlert"] [data-baseweb="button"] {
        background-color: #fff7e6 !important; 
        color: #333333 !important; 
        border-left-color: #ffb300 !important; 
    }
    div[data-testid="stAlert"] [data-baseweb="button"] svg {
        fill: #ffb300 !important;  
    }  
    /* CARD STYLE */
    .card {
        background-color: #f7f7f7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    /* UPLOAD BOX */
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
    /* LIMPIEZA */
    .main > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    header, .st-emotion-cache-18ni7ap {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    section[data-testid="stSidebar"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .block-container {
        padding-top: 0 !important; 
        margin-top: 1px !important;  
    }

</style>
""", unsafe_allow_html=True)

# ===================== 5. INTERFAZ =======================
WARM_PALETTE = ['#FFD700', '#FFB300', '#FF8C00', '#D2691E', '#8B4513']

# --- LÓGICA DE CARGA DEL LOGO (Base64) ---
import base64

logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Prod-AI(2).png') 

logo_html = ""
image_mimetype = "image/png" # Tipo MIME correcto

try:
    with open(logo_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    # Crea el tag IMG codificado con la data URI
    logo_html = f'<img src="data:{image_mimetype};base64,{encoded_string}" style="height:53px; vertical-align: middle; margin-right: 15px; border-radius: 2px;">'

except FileNotFoundError:
    # Muestra un mensaje de advertencia si la ruta sigue siendo incorrecta
    st.sidebar.warning(f"Error de ruta: No se encontró la imagen en '{logo_path}'.")
    logo_html = f'<span style="font-size: 24px; vertical-align: middle; margin-right: 15px;">❌</span>'
except Exception as e:
    # Captura otros errores (p. ej., problemas de permisos)
    st.sidebar.error(f"Error al cargar/codificar el logo: {e}")
    logo_html = f'<span style="font-size: 24px; vertical-align: middle; margin-right: 15px;">❌</span>'


# ------------ NAVBAR ------------
st.markdown(f"""
<div class='top-bar'>
    {logo_html}
    Prod.AI — Music Genre Detector & Stem Splitter 
</div>
""", unsafe_allow_html=True)

# ------------ LAYOUT -------------
# ... (el resto del código continúa)


# ------------ LAYOUT -------------
left, right = st.columns([2, 1], gap="large")
# Inicialización de todos los estados de sesión necesarios
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'stem_results' not in st.session_state:
    st.session_state.stem_results = None
if 'pred_genre' not in st.session_state:
    st.session_state.pred_genre = None
if 'producer_metrics' not in st.session_state:
    st.session_state.producer_metrics = None
if 'probs_dict' not in st.session_state:
    st.session_state.probs_dict = None
if 'times' not in st.session_state:
    st.session_state.times = None
if 'rms' not in st.session_state:
    st.session_state.rms = None


# ------------ LEFT PANEL (Carga y Análisis) -------------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Drop your audio file")

    audio = st.file_uploader(" ", type=["mp3","wav","m4a"], label_visibility="collapsed")

    # --- LÓGICA CLAVE PARA EVITAR RECARGAS PESADAS ---
    
    # Condición 1: Se subió un archivo y **AÚN NO** se han guardado los resultados en session_state
    if audio is not None and st.session_state.pred_genre is None:
        
        # Guardar archivo temporalmente para su procesamiento
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.type.split('/')[-1]}") as tmp_file:
            tmp_file.write(audio.read())
            audio_path = tmp_file.name
        
        # CÁLCULOS PESADOS (SÓLO AQUÍ)
        with st.spinner("Analyzing music and calculating metrics..."):
            pred_genre, probs_dict = predict_song(audio_path)
            times, rms = calculate_track_energy(audio_path)
            producer_metrics = calculate_producer_metrics(audio_path)

        # GUARDAR EN ESTADO DE SESIÓN
        st.session_state.audio_path = audio_path
        st.session_state.pred_genre = pred_genre
        st.session_state.probs_dict = probs_dict
        st.session_state.producer_metrics = producer_metrics
        st.session_state.times = times
        st.session_state.rms = rms
        st.session_state.stem_results = None # Resetear stems si el audio es nuevo
        
        # Forzar un rerun LIGERO para que el resto del script se cargue usando los datos guardados
        st.experimental_rerun() 


    # Condición 2: Ya existen datos en session_state (ya se calcularon, o se acaban de calcular)
    if st.session_state.pred_genre is not None:
        
        # Recuperar los datos del estado de sesión (¡ES INSTANTÁNEO!)
        pred_genre = st.session_state.pred_genre
        probs_dict = st.session_state.probs_dict
        producer_metrics = st.session_state.producer_metrics
        times = st.session_state.times
        rms = st.session_state.rms

        st.header(f"  Genre Detected: **{pred_genre}**")
        
        # 2. GRÁFICA DE DONUT Y CONTENEDOR DE MÉTRICAS
        col_genre, col_spacer, col_metrics = st.columns([1.5, 0.1, 1]) 
        
        with col_genre:
            st.markdown("##### Probabilidades")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie(list(probs_dict.values()), labels=list(probs_dict.keys()),
                    autopct='%1.1f%%', startangle=90,
                    colors=WARM_PALETTE[:len(probs_dict)])
            ax1.set_title("") 
            st.pyplot(fig1, use_container_width=True)
        
        # 3. ALTERNANCIA DE GRÁFICOS DE MÉTRICAS (Barras vs. Radar)
        with col_metrics:
            st.markdown("##### Metrics Visualization")
            
            # --- SELECTBOX PARA ALTERNAR VISTA (DESPLEGABLE) ---
            view_mode = st.selectbox(
                "Select View:",
                ("Key Metrics(Bars)", "Genre Comparison"),
                label_visibility="collapsed",
                key="metric_view"
            )
            
            # --- LÓGICA DE VISUALIZACIÓN ---
            
            if view_mode == "Key Metrics(Bars)":
                # --- VISTA DE BARRAS ---
                # NOTA: Aquí se usa el DataFrame con las CLAVES ORIGINALES del diccionario (RMS, Centroid, etc.)
                # Pero la columna 'Métrica' se debe mapear si quieres los nombres largos en las barras.
                
                # Para mostrar nombres más descriptivos en las barras, mapearemos las claves:
                metric_names = {
                    "RMS": "Energía RMS",
                    "Centroid": "Brillo (Centroide)",
                    "Lows": "Energía de Graves (<80Hz)",
                    "DR": "Rango Dinámico (DR)"
                }
                
                # Creamos el DataFrame usando las claves cortas, y luego mapeamos a nombres largos para la visualización.
                df_metrics = pd.DataFrame(
                    list(producer_metrics.items()), 
                    columns=['Clave', 'Valor'] # Cambiamos el nombre de la columna para evitar confusión
                )
                df_metrics['Métrica'] = df_metrics['Clave'].map(metric_names)

                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=df_metrics['Métrica'],
                            y=df_metrics['Valor'],
                            marker_color=WARM_PALETTE[:len(df_metrics)],
                            text=[f"{v:.1f}" for v in df_metrics['Valor']],
                            textposition='auto',
                            width=0.9 
                        )
                    ]
                )

                fig_bar.update_layout(
                    height=300, 
                    margin=dict(l=10, r=10, t=20, b=10),
                    plot_bgcolor="white", 
                    paper_bgcolor="white",
                    yaxis=dict(range=[0, 100], title="Relative Punctuation(0-100)"),
                    xaxis_title=None
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            elif view_mode == "Genre Comparison":
                # --- VISTA RADAR ---
                # Esta vista ahora funciona porque 'producer_metrics' usa las claves cortas (RMS, Centroid...)
                fig_radar = create_radar_chart(producer_metrics, pred_genre)
                st.plotly_chart(fig_radar, use_container_width=True)


        # 4. GRÁFICA DE ENERGÍA (RMS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=rms, mode="lines", line=dict(color="#ff9900", width=2)))
        mean_rms = np.mean(rms)
        fig.add_hline(y=mean_rms, line_dash="dash", line_color="#33AFFF", 
                    annotation_text=f"Average Energy: {mean_rms:.3f}", 
                    annotation_position="bottom right", annotation_font_size=10)
        fig.update_layout(
            height=380, 
            margin=dict(l=10, r=10, t=10, b=10), 
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="Time (s)", yaxis_title="Amplitud RMS", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
            
    # Si no hay datos analizados, mostrar el placeholder
    else:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.write("Drag & drop an audio file here") 
        st.markdown("</div>", unsafe_allow_html=True)
    
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
