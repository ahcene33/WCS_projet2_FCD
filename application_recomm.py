import streamlit as st
import pandas as pd
import pickle
from tmdb_api import get_movie_details_tmdb, has_synopsis_or_trailer
from omdb_api import get_omdb_synopsis
from enrichir import enrich_film_row
import base64
import numpy as np

import gdown
import os

def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)


tmdb_api_key = st.secrets["TMDB_API_KEY"]
omdb_api_key = st.secrets["OMDB_API_KEY"]

DEFAULT_POSTER = "None.png"
st.set_page_config(page_title="Application de recommandation de films", layout="wide")


# Fonction : fond d'écran fixe (page d'accueil)
def set_static_background(png_file):
    with open(png_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        div[data-testid="stApp"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Fonction : fond dynamique (après entrée)
def set_background_scroll(png_file):
    with open(png_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        div[data-testid="stApp"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center top;
            background-repeat: no-repeat;
            background-attachment: scroll;
            opacity: 0.9;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Contrôle de la page d'accueil
if "has_started" not in st.session_state:
    st.session_state.has_started = False

if not st.session_state.has_started:
    set_static_background("background2.png")  # FOND page accueil

    st.markdown(
        """
        <h1 style='text-align: center; color: white;'> Bienvenue sur F.C Data RecoMovies</h1>
        <p style='text-align: center; color: white; font-size: 20px;'>Découvrez des recommandations de films à partir de vos envies ou de grandes sagas cinématographiques.</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button("Entrer dans l'application", use_container_width=True):
        st.session_state.has_started = True

    st.stop()
else:
    set_background_scroll("background3.png")  # FOND dynamique pour l'app

@st.cache_data
def load_data():
    return pd.read_csv("data_clean.csv", parse_dates=['release_date'])


# Chargement des données
with st.spinner("Chargement des données..."):
    df_full = load_data()
    df = df_full.copy()


# Initialisation des variables de session
if "current_movie_id" not in st.session_state:
    st.session_state.current_movie_id = None
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

# Question initiale à l'utilisateur
choix_user = st.radio(
    "Vous avez une idée du film que vous souhaitez qu'on vous suggère en fonction ?",
    ("Oui", "Non")
)

if choix_user == "Non":
    st.subheader("Films les mieux notés par les spectateurs")

    df_top_films = df_full.copy()
    df_top_films = df_top_films[df_top_films['vote_count'] > 1000]
    df_top_films = df_top_films[df_top_films['vote_average'] >= 7.5]
    df_top_films["vote_score"] = df_top_films["vote_average"] * np.log1p(df_top_films["vote_count"])

    top_films = df_top_films.sort_values("vote_score", ascending=False).head(30)

    if st.button("Me surprendre avec un excellent film", key="btn_surprise_top_main"):
        film_random = top_films.sample(1).iloc[0]
        st.session_state.current_movie_id = film_random['id']
        st.rerun()

    cols = st.columns(3)
    for i, (_, film) in enumerate(top_films.iterrows()):
        with cols[i % 3]:
            st.image(film['poster_url'] if pd.notna(film['poster_url']) else DEFAULT_POSTER, width=150)
            if st.button(film['title'], key=f"top_main_film_{film['id']}"):
                st.session_state.current_movie_id = film['id']
                st.rerun()




#initialisation 

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

# IDs Drive à remplacer avec les tiens !
download_from_drive("1VkJzqcN3Fbu-sFnCbdqtRyEIfGPcETI6", "models/features_df.pkl")
download_from_drive("1v7SdiKqzw5hsDdsLXTVTpQ2tn3V7I2zn", "models/knn_model.pkl")


@st.cache_resource
def load_knn_model():
    with open('models/knn_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_features_df():
    with open('models/features_df.pkl', 'rb') as f:
        return pickle.load(f)
    
@st.cache_data
def enrich_film_row_cached(row, tmdb_api_key):
    return enrich_film_row(row, tmdb_api_key)


# Initialisation du film actif dans la session
def afficher_film_et_recommandations(movie_id):
    film_row = df[df['id'] == movie_id].iloc[0]
    with st.spinner("Chargement du film..."):
        film_row = enrich_film_row_cached(film_row, tmdb_api_key)

    st.subheader(f"{film_row['title']} ({film_row['release_date'].year})")
    col1, col2 = st.columns([1, 2])
    with col1:
        poster = film_row['poster_url'] if pd.notna(film_row['poster_url']) else DEFAULT_POSTER
        st.image(poster, width=250)
    with col2:
        synopsis_fallback, _ = get_movie_details_tmdb(film_row['id'], tmdb_api_key)
        st.markdown(f"**Résumé :** {synopsis_fallback}")
        st.markdown("**Tags :**")
        tags = film_row['genres'].split(' ') + [film_row['original_language'].upper()]
        if film_row.get('saga_name_clean') and film_row['saga_name_clean'] != 'Other':
            tags.append(f"Saga : {film_row['saga_name_clean']}")

    # Recommandations
    indices_recommandes = recommend_movie(movie_id)
    if indices_recommandes is not None:
        st.subheader("Recommandations similaires :")
        for idx in indices_recommandes[:5]:
            rec_film = df.iloc[idx]
            rec_movie_id = rec_film['id']
            st.image(rec_film['poster_url'] if pd.notna(rec_film['poster_url']) else DEFAULT_POSTER, width=150)
            if st.button(rec_film['title'], key=f"rec_{rec_movie_id}"):
                
                st.rerun()


st.subheader("Explorez nos sélections de films")

tab1, tab2, tab3, tab4 = st.tabs(["Blockbusters", "Classiques", "Arts & Essais", "Langues"])

with tab1:
    st.write("Films populaires récents (après 2000)")
    df_blockbusters = df[(df['release_date'].dt.year >= 2000) & (df['popularity'] > df['popularity'].quantile(0.90))]
    random_film = df_blockbusters.sample(1).iloc[0]
    if st.button("Voir un blockbuster au hasard ", key="btn_blockbuster"):
        st.session_state.current_movie_id = random_film['id']
        st.rerun()

with tab2:
    st.write("Classiques d'avant 1980, très appréciés")
    df_classics = df[(df['release_date'].dt.year < 1980) & (df['vote_average'] > 7.5)]
    if not df_classics.empty:
        random_film = df_classics.sample(1).iloc[0]
        if st.button("Voir un classique au hasard", key="btn_classic"):
            st.session_state.current_movie_id = random_film['id']
            st.rerun()


with tab3:
    st.write("Films d'art & essai (Documentary, Drama, etc.)")
    keywords = ['Documentary', 'Drama', 'Romance', 'History', 'Biography']
    mask_arts = df['genres'].str.contains('|'.join(keywords), case=False, na=False)
    df_arts = df[mask_arts & (df['vote_average'] > 7)]
    if not df_arts.empty:
        random_film = df_arts.sample(1).iloc[0]
        if st.button("Voir un film d’art & essai au hasard", key="btn_art"):
            st.session_state.current_movie_id = random_film['id']
            st.rerun()


with tab4:
    choix_langue = st.selectbox("Choisissez une langue", df['original_language'].unique())
    df_langue = df[df['original_language'] == choix_langue]
    if not df_langue.empty:
        random_film = df_langue.sample(1).iloc[0]
        if st.button(f"Voir un film en {choix_langue.upper()} au hasard", key="btn_langue"):
            st.session_state.current_movie_id = random_film['id']
            st.rerun()
    else:
        st.warning("Aucun film dans cette langue.")



with st.spinner("Chargement des vecteurs de films..."):
    features_df = load_features_df()

with st.spinner("Chargement du modèle de recommandation..."):
    knn_model = load_knn_model()

def recommend_movie(movie_id, n_recommendations=5):
    idx_list = df[df['id'] == movie_id].index.tolist()
    if not idx_list:
        st.error("Film ID non trouvé.")
        return []
    idx = idx_list[0]
    distances, indices = knn_model.kneighbors([features_df.iloc[idx].values], n_neighbors=n_recommendations + 1)
    rec_indices = indices.flatten()[1:]
    return rec_indices

with st.expander("Rechercher un film manuellement"):
    titre_input = st.text_input("Entrez un mot clé du titre :", key="search_main")

    if titre_input:
        resultats = df_full[df_full['original_title'].str.contains(titre_input, case=False, na=False)].sort_values("popularity", ascending=False).head(10)
        titres_possibles = resultats['original_title'].tolist()

        if titres_possibles:
            film_choisi = st.selectbox("Choisissez un film :", titres_possibles)
            if st.button("Lancer la recherche", key="btn_search"):
                film_row = resultats[resultats['original_title'] == film_choisi].iloc[0]
                st.session_state.current_movie_id = film_row['id']
                st.rerun()
        else:
            st.warning("Aucun film trouvé.")


if st.session_state.current_movie_id is not None:
    film_row = df[df['id'] == st.session_state.current_movie_id].iloc[0]
    movie_id = film_row['id']
    
    with st.spinner("Enrichissement du film sélectionné..."):
        film_row = enrich_film_row_cached(film_row, tmdb_api_key)

    st.subheader(f"{film_row['title']} ({film_row['release_date'].year})")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        poster = film_row['poster_url'] if pd.notna(film_row['poster_url']) else DEFAULT_POSTER
        st.image(poster, width=250)

    with col_right:
        synopsis_fallback, _ = get_movie_details_tmdb(film_row['id'], tmdb_api_key)
        st.markdown(f"**Résumé :** {synopsis_fallback}")
        
        st.markdown("**Tags :**")
        tags = []

        if pd.notna(film_row.get('genres')):
            tags += film_row['genres'].split(' ')
        if pd.notna(film_row.get('original_language')):
            tags.append(film_row['original_language'].upper())
        if film_row.get('saga_name_clean') and film_row['saga_name_clean'] != "Other":
            tags.append(f"Saga : {film_row['saga_name_clean']}")

        st.markdown(
    "<div>" + "".join(f"<span class='tag'>{tag}</span>" for tag in tags) + "</div>",
    unsafe_allow_html=True
)

        st.markdown(f"**Note :** {film_row['vote_average']}")
        st.markdown(f"**Popularité :** {int(film_row['popularity'])}")


    if st.button("Voir plus (Synopsis FR + Trailer)", key="main_voir_plus"):
        synopsis_fr, trailer_url = get_movie_details_tmdb(movie_id, tmdb_api_key)
        st.write("**Synopsis FR :**")
        st.write(synopsis_fr)
        if trailer_url:
            st.video(trailer_url)
        else:
            st.write("Trailer non disponible.")
        imdb_id = film_row['imdb_id']
        synopsis_omdb = get_omdb_synopsis(imdb_id, omdb_api_key)
        st.write("**Synopsis OMDb (via imdb_id) :**")
        st.write(synopsis_omdb)

    indices_recommandes = recommend_movie(movie_id)
    if indices_recommandes is not None:
        st.subheader("Recommandations similaires")
        reco_films = [df.iloc[idx] for idx in indices_recommandes[:5]]
        cols = st.columns(len(reco_films))

        for i, rec_film in enumerate(reco_films):
            rec_movie_id = rec_film['id']
            with cols[i]:
                with st.spinner(f"Chargement : {rec_film['title']}"):
                    rec_film = enrich_film_row(rec_film, tmdb_api_key)
                poster = rec_film['poster_url'] if pd.notna(rec_film['poster_url']) else DEFAULT_POSTER

                if st.button(f"{rec_film['title']}", key=f"img_{rec_movie_id}"):
                    st.session_state.current_movie_id = rec_movie_id
                    st.rerun()



                st.image(poster, width=140)
                st.markdown(f"**{rec_film['title']}**", unsafe_allow_html=True)
                st.caption(f"{rec_film['release_date'].year} | {rec_film['vote_average']}/10")



st.markdown("---")
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("logo_wild.png", width=50)
with col2:
    st.markdown("Cette application a été réalisée par **F.C Data, Ahcene K, Kamel T & Majed S**, Wild Code School 2025.")