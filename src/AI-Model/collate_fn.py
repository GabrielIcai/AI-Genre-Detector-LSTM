import torch
from collections import defaultdict
import torch
import re


def extract_song_name(image_path):
    match = re.match(r"(.*)_fragmento_\d+\.png", image_path)
    if match:
        return match.group(1)
    else:
        return None

# Para genero necesito agrupar fragmentos de las mismas canciones en un batch que no puedo hacer con un dataset normal. Con collate proceso y agrupo varios
# elementos individuales
def collate_fn(batch):
    grouped_by_song = defaultdict(list)

    # Agrupar los fragmentos por canción
    for img, add_feats, label, image_path in batch:
        if img is not None:
            song_name = extract_song_name(image_path)
            if song_name:
                grouped_by_song[song_name].append((img, add_feats, label))

    images = []
    additional_features = []
    labels = []

    for song_name, fragments in grouped_by_song.items():
        # Padding
        while len(fragments) % 3 != 0:
            fragments.append(
                (
                    torch.zeros_like(fragments[0][0]),
                    torch.zeros_like(fragments[0][1]),
                    torch.zeros_like(fragments[0][2]),
                )
            )
        # Crear ventanas de 3 fragmentos consecutivos
        for i in range(0, len(fragments), 3):
            song_images = []
            song_additional_features = []
            song_labels = []

            # Coger 3 fragmentos
            for j in range(3):
                img, add_feats, label = fragments[i + j]
                song_images.append(img)
                song_additional_features.append(add_feats)
                song_labels.append(label)
            
            song_labels = song_labels[0]  # Uso la etiqueta del primer fragmento
            labels.append(song_labels)
            
            # Convertir las listas de fragmentos en tensores
            images.append(torch.stack(song_images, dim=0))
            additional_features.append(torch.stack(song_additional_features, dim=0))

    # Convertir las listas en un solo tensor
    images = torch.stack(images, dim=0)  # (batch_size, 3, canales, altura, anchura)
    additional_features = torch.stack(
        additional_features, dim=0
    )  # (batch_size, 3, num_features)
    labels = torch.stack(labels, dim=0)  # (batch_size, 3, num_labels)

    return images, additional_features, labels


def collate_fn_s(batch):
    batch = [item for item in batch if item[0] is not None]

    grouped_by_song = defaultdict(list)

    for image, add_feats, label, song_id in batch:
        grouped_by_song[song_id].append((image, add_feats, label))

    images, additional_features, labels = [], [], []

    for song_id, fragments in grouped_by_song.items():
        print(f"Procesando canción {song_id} con {len(fragments)} fragmentos")
        
        if len(fragments) < 3:
            print(f"Advertencia: No hay suficientes fragmentos para la canción {song_id}, se omite")
            continue

        while len(fragments) % 3 != 0:
            fragments.pop() 

        for i in range(0, len(fragments), 3):
            song_images = torch.stack([fragments[i+j][0] for j in range(3)], dim=0)
            song_additional_features = torch.stack([fragments[i+j][1] for j in range(3)], dim=0)
            song_label = fragments[i][2] 

            images.append(song_images)
            additional_features.append(song_additional_features)
            labels.append(song_label)

    if len(images) == 0:
        print("Advertencia: No se han creado imágenes para este lote")
    else:
        print(f"Se han creado {len(images)} imágenes")

    if len(images) > 0:
        images = torch.stack(images, dim=0)
        additional_features = torch.stack(additional_features, dim=0)
        labels = torch.stack(labels, dim=0)

    return images, additional_features, labels

#PARA PREDICCIÓN#

def collate_fn_prediction(batch):
    grouped_by_song = defaultdict(list)

    # Agrupar fragmentos por canción usando el string song_id directamente
    for img, add_feats, song_id in batch:
        if img is not None:
            grouped_by_song[song_id].append((img, add_feats))

    images = []
    additional_features = []
    song_names = []

    for song_name, fragments in grouped_by_song.items():
        while len(fragments) % 3 != 0:
            fragments.append(
                (
                    torch.zeros_like(fragments[0][0]),  
                    torch.zeros_like(fragments[0][1]), 
                )
            )

        # Crear ventanas de 3 fragmentos consecutivos
        for i in range(0, len(fragments), 3):
            song_images = []
            song_additional_features = []

            for j in range(3):
                img, add_feats = fragments[i + j]
                song_images.append(img)
                song_additional_features.append(add_feats)

            song_names.append(song_name)
            images.append(torch.stack(song_images, dim=0))
            additional_features.append(torch.stack(song_additional_features, dim=0))

    # Convertir listas a tensores
    images = torch.stack(images, dim=0)
    additional_features = torch.stack(additional_features, dim=0)

    return images, additional_features, song_names
