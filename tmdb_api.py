import requests

def get_movie_details_tmdb(tmdb_id, tmdb_api_key, return_full_data=False):
    base_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=fr-FR&api_key={tmdb_api_key}"
    videos_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/videos?language=fr-FR&api_key={tmdb_api_key}"
    
    details = requests.get(base_url).json()
    
    if return_full_data:
        return details  # utilisé dans enrich.py

    videos = requests.get(videos_url).json()
    synopsis = details.get('overview', 'Synopsis non disponible.')
    
    trailer_url = None
    for video in videos.get('results', []):
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
            break
    
    return synopsis, trailer_url

def has_synopsis_or_trailer(tmdb_id, tmdb_api_key):
    synopsis, trailer_url = get_movie_details_tmdb(tmdb_id, tmdb_api_key)
    return (synopsis != 'Synopsis non disponible.') or (trailer_url is not None)
