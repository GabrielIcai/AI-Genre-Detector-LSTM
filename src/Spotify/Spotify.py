import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
))

import random

def get_recommended_tracks(predicted_genres_probabilities, total_tracks=3, market='US'):
    
    if not predicted_genres_probabilities:
        return []

    main_genre = max(predicted_genres_probabilities, key=predicted_genres_probabilities.get)

    try:
        final_query = f'genre:"{main_genre}"'

        # 🎯 Offset aleatorio entre 0 y 1000
        random_offset = random.randint(0, 1000)

        results = sp.search(
            q=final_query,
            type='track',
            limit=total_tracks,
            market=market,
            offset=random_offset
        )

        recommendations = []

        if results and results['tracks']['items']:
            for track in results['tracks']['items']:
                track_info = {
                    "name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "image": track['album']['images'][0]['url'] if track['album']['images'] else "https://via.placeholder.com/150"
                }
                recommendations.append(track_info)

        return recommendations

    except Exception as e:
        print("Error al obtener recomendaciones de Spotify:", e)
        return []
