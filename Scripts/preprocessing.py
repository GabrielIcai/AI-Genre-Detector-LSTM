import os
import sys

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.preprocessing import (
    load_data,
    normalize_columns,
    normalize_images)
import torch
from src.preprocessing.data_loader import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]

#Cargo las imagenes desde drive
data_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida1/dataset_genero_completo.csv"
base_path = "/content/drive/MyDrive/TFG/data/"
normalized_folder = "/content/drive/MyDrive/TFG/images/espectrogramas_normalizados/"

def main():
    # Preprocesado
    data = load_data(data_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]

    print(data.head(10))

    normalize_columns(data, columns)

    for img_path in data["Ruta"]:
        if not os.path.exists(img_path):
            print(f"Ruta no encontrada: {img_path}")
    print("Rutas comprobadas")
    
    normalize_images(data,normalized_folder)

if __name__ == "__main__":
    main()