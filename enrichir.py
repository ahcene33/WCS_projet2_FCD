import pandas as pd
from tmdb_api import get_movie_details_tmdb

def enrich_film_row(film_row, tmdb_api_key):
    """
    Complète les infos manquantes d’un film via l’API TMDb.
    Retourne la ligne mise à jour.
    """
    film_row = film_row.copy()
    try:
        tmdb_data = get_movie_details_tmdb(film_row['id'], tmdb_api_key, return_full_data=True)
        if not tmdb_data:
            return film_row

        # Champs à enrichir
        champs_a_remplir = ['release_date', 'runtime', 'vote_average', 'vote_count', 'revenue']
        for champ in champs_a_remplir:
            if champ in film_row and pd.isna(film_row[champ]):
                film_row[champ] = tmdb_data.get(champ)

        # Poster
        if pd.isna(film_row.get('poster_url')):
            poster_path = tmdb_data.get('poster_path')
            if poster_path:
                film_row.loc['poster_url'] = f"https://image.tmdb.org/t/p/w500{poster_path}"

        # Saga
        if 'saga_name_clean' in film_row and pd.isna(film_row['saga_name_clean']):
            collection = tmdb_data.get('belongs_to_collection')
            if collection:
                name = collection.get('name')
                film_row['saga_name'] = name
                film_row['saga_name_clean'] = name
            else:
                film_row['saga_name'] = 'Other'
                film_row['saga_name_clean'] = 'Other'

        return film_row

    except Exception as e:
        print(f"[enrich] Erreur enrichissement film {film_row.get('id')} : {e}")
        return film_row
