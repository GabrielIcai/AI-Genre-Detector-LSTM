import os
import sys
import json

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.preprocessing import (
    load_data,
    mean_std_image,
    normalize_columns,
)
import torch
from src.preprocessing.data_loader import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]

data_path = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados_emociones_estructura/dataset_emociones_secciones.csv"
base_path = "/content/drive/MyDrive/TFG/images/"
output_path = "/content/drive/MyDrive/TFG/mean_std_emotions.json"

columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]

def main():
    data = load_data(data_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]
    data["Ruta"] = data["Ruta"].str.replace("espectrogramas_salida_secciones_2", "espectrogramas_normalizados_emociones_estructura")

    print("Primeras filas del dataset:")
    data=data.head(30)
    normalize_columns(data, columns)
    print("Primeras filas después de normalización:")
    print(data.head(4))

    # Verificar rutas
    for img_path in data["Ruta"]:
        if not os.path.exists(img_path):
            print(f"Ruta no encontrada: {img_path}")
    print("Rutas comprobadas")

    mean, std = mean_std_image(data)

    #JSON
    mean_std_data = {"mean": mean.tolist(), "std": std.tolist()}
    with open(output_path, "w") as f:
        json.dump(mean_std_data, f)
    print(f"Media y desviación estándar guardadas en: {output_path}")

if __name__ == "__main__":
    main()