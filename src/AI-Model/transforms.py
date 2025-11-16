import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch
import gc



def normalize_columns(data, columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])


def normalize_images(data,normalized_folder):
    os.makedirs(normalized_folder, exist_ok=True)
    normalized_images = []
    for i, img_path in enumerate(data["Ruta"]):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error al leer la imagen: {img_path}")
                continue
            img = img / 255.0
            filename = os.path.basename(img_path)
            normalized_path = os.path.join(normalized_folder, filename)

            # Guardar la imagen normalizada
            cv2.imwrite(normalized_path, (img * 255).astype('uint8'))

            if (i + 1) % 10 == 0:
                print(f"Procesadas {i + 1}/{len(data)} imágenes")
        except Exception as e:
            print(f"Error normalizing image {img_path}: {e}")
            if (i + 1) % 10 == 0:
                print(f"Procesadas {i + 1}/{len(data)} imágenes")
        except Exception as e:
            print(f"Error normalizing image {img_path}: {e}")
    return np.array(normalized_images)


def mean_std_image(data, batch_size=100):
    transform = transforms.ToTensor()
    total_mean = torch.zeros(3)
    total_std = torch.zeros(3)
    total_pixels = 0

    for i in tqdm(range(0, len(data), batch_size), desc="Calculando mean/std"):
        batch_paths = data["Ruta"][i:i + batch_size]
        batch_means = []
        batch_stds = []
        batch_pixels = 0

        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                tensor_image = transform(image)
                batch_pixels += tensor_image.shape[1] * tensor_image.shape[2] 
                batch_means.append(tensor_image.mean([1, 2]))
                batch_stds.append(tensor_image.std([1, 2]))
            except Exception as e:
                print(f"Error en {img_path}: {e}")

        if batch_means:
            batch_means = torch.stack(batch_means).mean(0)
            batch_stds = torch.stack(batch_stds).mean(0)

            # GLOBAL
            total_mean += batch_means * batch_pixels
            total_std += batch_stds * batch_pixels
            total_pixels += batch_pixels

        # Liberar memoria
        gc.collect()

    # Calcular media y std globales
    global_mean = total_mean / total_pixels
    global_std = total_std / total_pixels

    return global_mean, global_std
