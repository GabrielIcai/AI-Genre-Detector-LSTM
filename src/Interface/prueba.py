import streamlit as st
import torch
import librosa
import soundfile as sf
import plotly.graph_objects as go
import tempfile
from demucs.pretrained import get_model
from demucs.apply import apply_model
import numpy as np

def plot_waveform(audio, sr, title):
    t = np.linspace(0, len(audio) / sr, len(audio))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=audio, mode='lines'))
    fig.update_layout(title=title, xaxis_title='Tiempo (s)', yaxis_title='Amplitud')
    return fig

st.title("🎧 Separador de Voces y Música (Demucs)")

uploaded_file = st.file_uploader("Sube un archivo .mp3 o .wav", type=["mp3", "wav"])

if uploaded_file:
    # Guardar temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    # Cargar audio
    y, sr = librosa.load(input_path, sr=44100, mono=False)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)  # convertir a [canales, samples]

    wav = torch.from_numpy(y).float().unsqueeze(0)  # [1, canales, samples]

    # Separación
    with st.spinner("Separando pistas con Demucs..."):
        model = get_model("htdemucs").to("cpu")
        model.eval()
        sources = apply_model(model, wav, device="cpu")[0]  # [stems, channels, samples]

    # Demucs hm devuelve: [0:drums,1:bass,2:other,3:vocals]
    vocals = sources[3].mean(axis=0).numpy()
    accompaniment = (sources[0] + sources[1] + sources[2]).mean(axis=0).numpy()

    # Guardar archivos temporales
    vocals_path = tempfile.mktemp(suffix=".wav")
    music_path = tempfile.mktemp(suffix=".wav")
    sf.write(vocals_path, vocals, sr)
    sf.write(music_path, accompaniment, sr)

    # Mostrar y reproducir stems
    #st.subheader("🎤 Voces")
    #st.plotly_chart(plot_waveform(vocals, sr, "Voces"), use_container_width=True)
    #st.audio(vocals_path)
#
    #st.subheader("🎼 Acompañamiento (Música)")
    #st.plotly_chart(plot_waveform(accompaniment, sr, "Música"), use_container_width=True)
    #st.audio(music_path)

    # Descarga
    st.download_button("⬇️ Descargar Voces", open(vocals_path, "rb"), "vocals.wav")
    st.download_button("⬇️ Descargar Música", open(music_path, "rb"), "music.wav")

