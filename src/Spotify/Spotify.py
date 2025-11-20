# Contenido de '../Spotify/Spotify.py'

import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

# 1. Función para inicializar Spotify (con credenciales pasadas)
def get_spotify_client(client_id, client_secret):
    """Inicializa y devuelve el cliente de Spotify."""
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        ))
        return sp
    except Exception as e:
        print(f"Fallo al inicializar SpotifyClient: {e}")
        return None # Devuelve None si falla

def get_recommended_tracks(predicted_genres_probabilities, total_tracks=3, market='US', sp_client=None):
    # ⚠️ Acepta el cliente de Spotify como argumento
    
    if sp_client is None:
        print("Error: El cliente de Spotify no está inicializado.")
        return []

    if not predicted_genres_probabilities:
        return []
    
    # 2. Usa el cliente sp_client en lugar de la variable global sp
    main_genre = max(predicted_genres_probabilities, key=predicted_genres_probabilities.get)
    
    try:
        final_query = f'genre:"{main_genre}"'
        random_offset = random.randint(0, 1000)

        results = sp_client.search(  # ⬅️ Cambio: Usar sp_client
            q=final_query,
            type='track',
            limit=total_tracks,
            market=market,
            offset=random_offset
        )

        # ... (el resto de la lógica de resultados sigue igual) ...
        recommendations = []
        if results and results['tracks']['items']:
            for track in results['tracks']['items']:
                # ... (resto de tu lógica de extracción de datos) ...
                track_info = {
                    "name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "image": track['album']['images'][0]['url'] if track['album']['images'] else "https://via.placeholder.com/150",
                    "artists": [{"name": track['artists'][0]['name']}] # Aseguramos que la estructura sea la que espera tu interfaz
                }
                recommendations.append(track_info)
        return recommendations

    except Exception as e:
        print("Error al obtener recomendaciones de Spotify:", e)
        return []
