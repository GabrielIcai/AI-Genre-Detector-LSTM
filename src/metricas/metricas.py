import librosa
import numpy as np
import os
import soundfile as sf
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, lfilter
# Añade esta función auxiliar DENTRO de tu archivo '../Metricas/metrica.py'

def bandpass_filter(data, lowcut, highcut, sr, order=5):
    """Implementa un filtro digital Butterworth de paso de banda."""
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    # 1. Diseñar el filtro (Butterworth)
    b, a = butter(order, [low, high], btype='band')
    # 2. Aplicar el filtro (lfilter)
    y = lfilter(b, a, data)
    return y

def calculate_producer_metrics(audio_path):
    # ... (el resto de tu código de carga y otras métricas) ...

    # --- 3. Energía de Graves (80Hz) ---
    # 🚨 REEMPLAZO DEL CÓDIGO CON ERROR
    y_bass = bandpass_filter(y, lowcut=20, highcut=80, sr=sr)
    
    # ... (el resto del cálculo se mantiene igual) ...
    bass_rms_linear = librosa.feature.rms(y=y_bass)[0]
    bass_energy_db = 20 * np.log10(np.mean(bass_rms_linear) + 1e-6)
    bass_energy_scaled = np.clip(bass_energy_db + 60, 0, 60) # Escala similar a RMS

    # ... (el resto de la función) ...

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
    y_bass = bandpass_filter(y, lowcut=20, highcut=80, sr=sr)
    bass_rms_linear = librosa.feature.rms(y=y_bass)[0]
    bass_energy_db = 20 * np.log10(np.mean(bass_rms_linear) + 1e-6)
    bass_energy_scaled = np.clip(bass_energy_db + 60, 0, 60) # Escala similar a RMS# Escala de 0 a 60

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
