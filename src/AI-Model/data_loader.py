import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(data_path):
    data = pd.read_csv(data_path, encoding="latin-1")
    return data

#0.8 de train y 0.2 de test
def split_dataset(data, test_size=0.2, random_state=42):
    train_data, val_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    return train_data, val_data

import librosa
import numpy as np
import pandas as pd

def load_data_from_audio(file_path, sample_rate=22050, hop_length=512, n_fft=2048):
    """
    Extrae características del archivo de audio dado y devuelve un DataFrame
    similar al que usabas en el CSV original.
    """
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

    # --- Features principales ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    mfccs_mean = mfccs.mean(axis=1)

    # --- Crear DataFrame similar al CSV ---
    data = pd.DataFrame({
        "Spectral Centroid": spectral_centroid,
        "Spectral Bandwidth": spectral_bandwidth,
        "Spectral Roll-off": spectral_rolloff,
        "Zero Crossing Rate": zero_crossing_rate,
        "RMS": rms
    })

    # Añadir MFCCs como columnas
    for i in range(12):
        data[f"MFCC_{i+1}"] = mfccs[i, :len(data)]

    # Si quieres mantener compatibilidad con CustomDataset:
    data["Ruta"] = file_path
    data["Song ID"] = os.path.basename(file_path).split('.')[0]

    return data

def normalize_columns(df, columns_to_normalize, mean_values=None, std_values=None):
    """
    Normaliza columnas según medias y desviaciones fijas (usadas en entrenamiento).
    """
    if mean_values is not None and std_values is not None:
        for i, col in enumerate(columns_to_normalize):
            if col in df.columns:
                df[col] = (df[col] - mean_values[i]) / std_values[i]
    else:
        # Fallback a normalización estándar si no se pasan medias fijas
        for col in columns_to_normalize:
            if col in df.columns:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
