import os
import torch
import librosa
import numpy as np
from torch.utils.data import DataLoader
from genre_model import CRNN
from collate_fn import collate_fn_prediction
from custom_dataset import c_transform
from PIL import Image

# --- CONFIGURACIÓN ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Final model/best_CRNN_genre_5_2.pth"
#MODEL_PATH = "Final model/CRNN_5_genre_2.pth"
CLASS_NAMES = ["Ambient", "Deep House", "Techno", "Trance", "Progressive House"]
MEAN = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
STD = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
DURACION_FRAGMENTO = 5  # segundos

# --- FUNCIONES AUXILIARES ---
def extract_features(y, sr):
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    mean_abs_amplitude = np.mean(np.abs(y))
    crest_factor = np.max(np.abs(y)) / rms if rms != 0 else 0
    std_amplitude = np.std(y)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr).mean()
    try:
        vad = librosa.effects.split(y=y, top_db=30)
        vad_result = 1 if len(vad) > 0 else 0
    except Exception:
        vad_result = 0
    try:
        spectrogram = np.abs(librosa.stft(y))
        spectral_variation = np.std(spectrogram, axis=1).mean()
    except Exception:
        spectral_variation = 0
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    except Exception:
        tempo = 0
    return np.array([
        rms, zcr, mean_abs_amplitude, crest_factor, std_amplitude,
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        spectral_flux, vad_result, spectral_variation, tempo
    ], dtype=np.float32)


def generate_spectrogram_image(fragmento, sr):
    import cv2
    espectrograma = librosa.feature.melspectrogram(y=fragmento, sr=sr, n_mels=128, fmax=8000)
    espectrograma_db = librosa.power_to_db(espectrograma, ref=np.max)
    espectrograma_norm = (espectrograma_db - np.min(espectrograma_db)) / (np.max(espectrograma_db) - np.min(espectrograma_db))
    espectrograma_norm = cv2.resize(espectrograma_norm, (512, 512))
    espectrograma_norm = np.expand_dims(espectrograma_norm, axis=-1)
    return espectrograma_norm


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, transform=None, duration=5):
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.duration_samples = duration * self.sr
        self.num_fragments = len(self.y) // self.duration_samples
        self.transform = transform
        self.samples = []

        for i in range(self.num_fragments):
            inicio = i * self.duration_samples
            fin = inicio + self.duration_samples
            fragmento = self.y[inicio:fin]
            img = generate_spectrogram_image(fragmento, self.sr)
            feats = extract_features(fragmento, self.sr)
            self.samples.append((img, feats))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, feats = self.samples[idx]
        img = Image.fromarray((img[:, :, 0] * 255).astype(np.uint8)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(feats, dtype=torch.float32), 0


def load_model():
    model = CRNN(num_classes=len(CLASS_NAMES), additional_features_dim=12, hidden_size=256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# --- PREDICCIÓN (sin gráficas) ---
def predict_song(audio_path):
    model = load_model()
    transform = c_transform(MEAN, STD)

    dataset = AudioDataset(audio_path, transform=transform, duration=DURACION_FRAGMENTO)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_prediction)

    all_probs = []
    with torch.no_grad():
        for images, feats, _ in loader:
            images, feats = images.to(DEVICE), feats.to(DEVICE)
            outputs = model(images, feats)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    avg_probs = np.mean(all_probs, axis=0)
    probs_dict = {genre: float(p) for genre, p in zip(CLASS_NAMES, avg_probs)}
    pred_genre = max(probs_dict, key=probs_dict.get)

    return pred_genre, probs_dict
