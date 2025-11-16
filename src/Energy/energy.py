
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def calculate_track_energy(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=512)
    return times, rms
