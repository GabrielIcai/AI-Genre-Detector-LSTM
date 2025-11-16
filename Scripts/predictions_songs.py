import torch
import pandas as pd
import numpy as np
import os
import seaborn as sns
import sys
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_path not in sys.path:
    sys.path.append(repo_path)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from src.preprocessing import CustomDataset_s, normalize_columns, load_data, c_transform,CustomDataset
from src.training import collate_fn_s,collate_fn
from src.models.genre_model import CRNN, CNN_LSTM_genre

#F
# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
for i in range (3,52):
    # Rutas
    base_path = "/content/drive/MyDrive/TFG/data/"
    model_path = "/content/drive/MyDrive/TFG/models/best_CRNN_genre_5.pth"
    csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_test.csv"
    output_csv_path = "/content/drive/MyDrive/TFG/predicciones_canciones_CRNN_5_generos.csv"

    # Normalización
    mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
    std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
    columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
    num_classes = 5
    class_names = [ "Ambient", "Deep House", "Techno", "Trance", "Progressive House"]

    model = CRNN(num_classes=num_classes, additional_features_dim=12, hidden_size=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Cargar los datos
    data = load_data(csv_path)
    data["Ruta"] = data["Ruta"].str.replace("\\", "/")
    data["Ruta"] = base_path + data["Ruta"]
    normalize_columns(data, columns)
    class_counts = data[[ "Ambient", "Deep House", "Techno", "Trance", "Progressive House"]].sum()
    class_names = ["Ambient", "Deep House", "Techno", "Trance", "Progressive House"]

    data = data[data["Song ID"] == f"song{i}"]

    # Verificar rutas
    for img_path in data["Ruta"]:
        if not os.path.exists(img_path):
            print(f"Ruta no encontrada: {img_path}")

    test_transform = c_transform(mean, std)

    # Asegúrate de que el CustomDataset incluya el Song ID
    test_dataset = CustomDataset(data, base_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=2, pin_memory=True)

    all_song_ids = np.repeat(data["Song ID"].values, 1)

    # Listas para almacenar los resultados
    all_preds = []
    all_labels = []
    all_probabilities = []

    # Realizar la inferencia
    with torch.no_grad():
        for i, (images, additional_features, labels) in enumerate(test_loader):
            images = images.to(device)
            additional_features = additional_features.to(device)
            labels = labels.to(device)

            outputs = model(images, additional_features)
            preds = torch.argmax(outputs, dim=1)
            labels_grouped = torch.argmax(labels, dim=1)
            probabilities = torch.softmax(outputs, dim=1)

            all_probabilities.extend(probabilities.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_grouped.cpu().numpy())

    # Emparejar Song ID con las predicciones
    all_song_ids = all_song_ids[:len(all_preds)]  # Ajustar longitud en caso de padding

    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame({
        'Song ID': all_song_ids,
        'Real Label': all_labels,
        'Predicted Label': all_preds,
        'Probabilities': all_probabilities
    })

    # Convertir etiquetas numéricas a nombres de clases
    results_df['Real Label'] = results_df['Real Label'].apply(lambda x: class_names[x])
    results_df['Predicted Label'] = results_df['Predicted Label'].apply(lambda x: class_names[x])

    # Guardar en CSV
    results_df.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

    # Mostrar las primeras filas
    print(results_df.head())