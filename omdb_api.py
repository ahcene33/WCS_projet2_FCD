import requests


def get_omdb_synopsis(imdb_id, omdb_api_key, language='fr'):
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={omdb_api_key}&plot=full"
    
    response = requests.get(url)
    data = response.json()
    
    # OMDb ne garantit pas la langue, mais parfois en français on trouve des plots traduits
    # on va renvoyer le plot si dispo
    plot = data.get('Plot', 'Synopsis non disponible.')
    
    return plot
