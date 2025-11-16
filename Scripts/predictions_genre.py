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
from src.preprocessing import CustomDataset, normalize_columns, load_data, c_transform
from src.training import collate_fn
from src.models.genre_model import CRNN, CNN_LSTM_genre

##PREDICCION POR FRAGMENTO##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
base_path = "/content/drive/MyDrive/TFG/data/"
model_path = "/content/drive/MyDrive/TFG/models/best_CRNN_genre_5.pth"
csv_path = "/content/drive/MyDrive/TFG/data/espectrogramas_salida_test/dataset_test.csv"
mean = [0.676956295967102, 0.2529653012752533, 0.4388839304447174]
std = [0.21755781769752502, 0.15407244861125946, 0.07557372003793716]
columns = ["Spectral Centroid", "Spectral Bandwidth", "Spectral Roll-off"]
hidden_size = 256
additional_features_dim = 12
num_classes = 6

model = CRNN(num_classes=num_classes, additional_features_dim=12, hidden_size=256)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

data = load_data(csv_path)
print(data.head(120))
data["Ruta"] = data["Ruta"].str.replace("\\", "/")
data["Ruta"] = base_path + data["Ruta"]
normalize_columns(data, columns)
class_counts = data[[ "Ambient", "Deep House", "Techno", "Trance","Progressive House"]].sum()
class_names = ["Ambient", "Deep House", "Techno", "Trance", "Progressive House"]

# Mostrar el conteo por clase
print("Distribución de clases en el conjunto de datos:")
print(class_counts)


# Verificar rutas
for img_path in data["Ruta"]:
    if not os.path.exists(img_path):
        print(f"Ruta no encontrada: {img_path}")

test_transform = c_transform(mean, std)

test_dataset = CustomDataset(data, base_path, transform=test_transform)
test_loader = DataLoader( test_dataset, batch_size=128, collate_fn=collate_fn, shuffle=False, num_workers=2, pin_memory=True)

# Después de completar la inferencia
all_preds = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for images, additional_features, labels in test_loader:
        images =images.to(device)
        additional_features =additional_features.to(device)
        labels =labels.to(device)

        outputs =model(images, additional_features)
        preds =torch.argmax(outputs, dim=1)
        labels_grouped =torch.argmax(labels, dim=1)
        probabilities =torch.softmax(outputs, dim=1)

        all_probabilities.extend(probabilities.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_grouped.cpu().numpy())

print("\nAnálisis de distribución de etiquetas reales y predicciones:")
real_counts = pd.Series(all_labels).value_counts()
pred_counts = pd.Series(all_preds).value_counts()

real_counts = real_counts.reindex(range(num_classes), fill_value=0)
pred_counts = pred_counts.reindex(range(num_classes), fill_value=0)

for i, class_name in enumerate(class_names):
    print(f"Clase '{class_name}':")
    print(f"  Etiquetas reales: {real_counts[i]}")
    print(f"  Predicciones: {pred_counts[i]}")

# Generar matriz de confusión
conf_matrix = confusion_matrix(all_labels, all_preds)
print("\nMatriz de confusión:")
print(conf_matrix)

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(all_labels, all_preds))

# Visualizar matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas Reales")
plt.title("Matriz de Confusión")

image_path = "/content/drive/MyDrive/TFG/matriz_confusion_5_generos_CRNN.png"
plt.savefig(image_path)
plt.close()

# Ejemplo de probabilidades
example_idx = 2  
probabilities = all_probabilities[example_idx]

print(f"\nProbabilidades para el ejemplo {example_idx}:")
print(f"Probabilidades: {probabilities}")
print(f"Predicción: {all_preds[example_idx]}")
print(f"Etiqueta real: {all_labels[example_idx]}")
