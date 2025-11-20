import librosa
import numpy as np
import os
import soundfile as sf
import pandas as pd
import plotly.graph_objects as go


def calculate_producer_metrics(audio_path):
    """
    Calcula 4 métricas clave de producción musical utilizando librerías existentes.
    """
    if not os.path.exists(audio_path):

        return {
            "Energía RMS": 0.0,
            "Brillo (Centroide)": 0.0,
            "Energía de Graves (<80Hz)": 0.0,
            "Rango Dinámico (DR)": 0.0
        }

    try:
        # Usar sf.read para leer el archivo de audio.
        y, sr = sf.read(audio_path)
        
        # Conversión a mono y re-sampleo a 22050 Hz (eficiencia y estandarización)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
            
    except Exception as e:
        print(f"Error al leer o procesar el audio: {e}")
        return {
            "Energía RMS": 0.0,
            "Brillo (Centroide)": 0.0,
            "Energía de Graves (<80Hz)": 0.0,
            "Rango Dinámico (DR)": 0.0
        }

    # --- Cálculo de Métricas (usando numpy y librosa) ---
    
    # 1. Energía RMS (Loudness)
    rms_linear = librosa.feature.rms(y=y)[0]
    rms_db = 20 * np.log10(np.mean(rms_linear) + 1e-6)
    rms_scaled = np.clip(rms_db + 60, 0, 60) # Escala de 0 a 60

    # 2. Centroide Espectral (Brillo)
    centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    centroid_scaled = np.clip(centroid_mean / 50, 0, 100) # Escala de 0 a 100

    # 3. Energía de Graves (<80Hz)
    y_bass = librosa.effects.bandpass(y, sr=sr, freq_min=20, freq_max=80)
    bass_rms_linear = librosa.feature.rms(y=y_bass)[0]
    bass_energy_db = 20 * np.log10(np.mean(bass_rms_linear) + 1e-6)
    bass_energy_scaled = np.clip(bass_energy_db + 60, 0, 60) # Escala de 0 a 60

    # 4. Rango Dinámico (DR)
    peak_amplitude = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak_amplitude + 1e-6)
    dynamic_range = peak_db - rms_db
    dynamic_range_scaled = np.clip(dynamic_range * 5, 0, 100) # Escala de 0 a 100

    return {
        "Energía RMS": rms_scaled,
        "Brillo (Centroide)": centroid_scaled,
        "Energía de Graves (<80Hz)": bass_energy_scaled,
        "Rango Dinámico (DR)": dynamic_range_scaled
    }