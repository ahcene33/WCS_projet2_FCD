import streamlit as st
import pandas as pd

# config de la page
st.set_page_config(page_title="Application de recommandation de films", layout="wide")

#importation du dataframe clean
data_tmdb = pd.read_csv("data_tmdb_clean_reduced.csv")  
films = data_tmdb.copy()
films['release_date'] = pd.to_datetime(films['release_date'], errors='coerce')

#_______________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________

# recommandation :
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#implélementation du modèle de maching learning et le fit

# nettoyage de la colonne text_features
# on force tout en string + on remplace les nan/None par string vide
films['text_features'] = films['text_features'].fillna('').astype(str)


#initialiser le vectorizer 
tfidf = TfidfVectorizer(stop_words='english', max_features=6000)

#l'appliquer à la colonne 'text_features' et la création de la matrice
tfidf_matrix =tfidf.fit_transform(films['text_features'])

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=300, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# calculer la similarité cosinus entre tous les films : modèle KNN sur IF IDF

from sklearn.neighbors import NearestNeighbors

# combien de voisin on vise ?
K = 10  # nombre de voisins

# initialiser le modèle KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tfidf_reduced)

# on prépare film_titles pour plus tard
film_titles = films['original_title'].str.strip().str.lower().tolist()



#créer la série indices qui reprend tous les titles de films en miniscules et en enlevant les espaces invisibles avec strip()

indices = pd.Series(films.index, index=films['original_title'].str.strip().str.lower())

# créationde la fonction recommandation_film 

# création de la fonction recommandation_film 

def recommandation_film(title, n=7) :

    title = title.strip().lower()

    if title not in indices :
        st.error(f"Film non trouvé dans la base de données. Veuillez vérifier l'orthographe.") 
        return None

    index_du_film = indices[title] # on recupere l'indice de film depuis son title de la datafram

    # --- Affichage du film sélectionné ---
    film_selectionne = films.iloc[index_du_film]
    st.subheader("Film sélectionné :")
    st.image(film_selectionne['poster_url'], width=200, caption=film_selectionne['title'])
    st.write(f"Genre : {film_selectionne['genres']}")
    st.write(f"Année de sortie : {film_selectionne['release_date'].year}")
    st.write(f"Résumé : {film_selectionne['overview']}")

    # On utilise knn pour obtenir les voisins
    distances, indices_neighbors = knn.kneighbors(tfidf_reduced[index_du_film].reshape(1, -1), n_neighbors=n+1)


    # On ignore le premier (le film lui-même)
    films_similaires = list(zip(indices_neighbors.flatten()[1:], 1 - distances.flatten()[1:]))

    indices_recommandes = [film[0] for film in films_similaires]
 
    return indices_recommandes

#_______________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________

# on récupére les titres disponibles dans la catégorie actuelle
titres_possibles = films['original_title'].sort_values().tolist()

# on met un sélecteur avec autocomplétion
titre_saisi = st.selectbox("Entrez un titre de film, et nous vous donnons des des recommandations de films en fonction :", \
                           options=titres_possibles)


if titre_saisi:
    indices_recommandes = recommandation_film(titre_saisi)

    if indices_recommandes:
        st.subheader("Recommandations similaires :")
        for idx in indices_recommandes:
            film = films.iloc[idx]
            st.image(film['poster_url'], width=150, caption=film['title'])
            st.write(f"Genre : {film['genres']}")
            st.write(f"Année de sortie : {film['release_date'].year}")
            st.write(f"Résumé : {film['overview']}")
            st.markdown("---")

#_______________________________________________________________________________________________________________________________
# Pied de page
st.markdown("---")  # ligne de séparation
col1, col2 = st.columns([0.1, 0.9])

with col1:
    st.image("logo_wild.png", width=50)

with col2:
    st.markdown("Cette application a été réalisée par **F.C Data, Ahcene K, Kamel T & Majed S**, Wild Code School 2025.")
    #test
