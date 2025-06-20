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

st.set_page_config(page_title="Application de recommandation de films", layout="wide")

def download_from_drive(file_id, output_path):
    if os.path.exists(output_path):
        return  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

@st.cache_resource
def run_once():
    download_from_drive("1LKJ9j2xz_J0sSpoiz-kWEppyN07-ZVFo", "models/features_df.csv")
    download_from_drive("1xqEXPXyGW-vlxgpq2h6AcOUHdcHCRSrK", "models/knn_model_data.csv")

run_once()


tmdb_api_key = st.secrets["TMDB_API_KEY"]
omdb_api_key = st.secrets["OMDB_API_KEY"]

DEFAULT_POSTER = "None.png"


# recomm 
def afficher_film(film_row):
    st.subheader(f"{film_row['title']} ({film_row['release_date'].year})")
    col1, col2 = st.columns([1, 2])
    with col1:
        poster = film_row['poster_url'] if pd.notna(film_row['poster_url']) else DEFAULT_POSTER
        st.image(poster, width=250)
    with col2:
        synopsis_fallback, _ = get_movie_details_tmdb(film_row['id'], tmdb_api_key)
        st.markdown(f"**Résumé :** {synopsis_fallback}")
        st.markdown("**Tags :**")
        afficher_tags(film_row)

        st.markdown(f"**Note :** {film_row['vote_average']}")
        st.markdown(f"**Popularité :** {int(film_row['popularity'])}")

def afficher_tags(film_row):
    tags = film_row['genres'].split(' ') + [film_row['original_language'].upper()]
    if film_row.get('saga_name_clean') and film_row['saga_name_clean'] != 'Other':
        tags.append(f"Saga : {film_row['saga_name_clean']}")
    st.markdown(
        "<div>" + "".join(f"<span class='tag'>{tag}</span>" for tag in tags) + "</div>",
        unsafe_allow_html=True
    )

def afficher_film_aleatoire(df_filtre, bouton_label, bouton_key):
    if not df_filtre.empty:
        film = df_filtre.sample(1).iloc[0]
        if st.button(bouton_label, key=bouton_key):
            st.session_state.current_movie_id = film['id']
            st.rerun()


if "film_cache" not in st.session_state:
    st.session_state.film_cache = {}

@st.cache_data(show_spinner=False)
def get_enriched_film_cached(movie_id, film_row, api_key):
    return enrich_film_row(film_row, api_key)

def get_enriched_film(film_row):
    movie_id = film_row["id"]
    if movie_id in st.session_state.film_cache:
        return st.session_state.film_cache[movie_id]
    enriched = get_enriched_film_cached(movie_id, film_row, tmdb_api_key)
    st.session_state.film_cache[movie_id] = enriched
    return enriched


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

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("data_clean.csv", parse_dates=['release_date'])


@st.cache_resource
def load_knn_model():
    with open('models/knn_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_features_df():
    return pd.read_csv("models/features_df.csv")

with st.spinner("Chargement des données..."):
    df = load_data()


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
    st.subheader("20 films très bien notés à découvrir")

    # Top films filtrés
    df_top_films = df[(df['vote_count'] > 1000) & (df['vote_average'] >= 7.5)].copy()
    df_top_films["vote_score"] = df_top_films["vote_average"] * np.log1p(df_top_films["vote_count"])

    # Prendre 20 films aléatoires parmi les top
    top_20_random = df_top_films.sample(n=20)

    # Affichage par lignes de 4
    cols = st.columns(4)
    for i, (_, film) in enumerate(top_20_random.iterrows()):
        with cols[i % 4]:
            st.image(film['poster_url'] if pd.notna(film['poster_url']) else DEFAULT_POSTER, width=140)
            st.caption(film['title'])
            if st.button("Voir", key=f"btn_top20_{film['id']}"):
                st.session_state.current_movie_id = film['id']
                st.rerun()


# Initialisation du film actif dans la session
def afficher_film_et_recommandations(movie_id):
    film_row = df[df['id'] == movie_id].iloc[0]
    with st.spinner("Chargement du film..."):
        film_row = get_enriched_film(film_row)

    afficher_film(film_row)  # 👈 appel unique et propre

    indices_recommandes = recommend_movie(movie_id)
    if indices_recommandes is not None:
        st.subheader("Recommandations similaires :")
        for idx in indices_recommandes[:5]:
            rec_film = df.iloc[idx]
            rec_movie_id = rec_film['id']
            st.image(rec_film['poster_url'] if pd.notna(rec_film['poster_url']) else DEFAULT_POSTER, width=150)
            if st.button(rec_film['title'], key=f"rec_{rec_movie_id}"):
                st.session_state.current_movie_id = rec_movie_id
                st.rerun()



st.subheader("Explorez nos sélections de films")

tab1, tab2, tab3, tab4 = st.tabs(["Blockbusters", "Classiques", "Arts & Essais", "Langues"])

with tab1:
    st.write("Films populaires récents (après 2000)")
    df_blockbusters = df[(df['release_date'].dt.year >= 2000) & (df['popularity'] > df['popularity'].quantile(0.90))]
    afficher_film_aleatoire(df_blockbusters, "Voir un blockbuster au hasard", "btn_blockbuster")


with tab2:
    st.write("Classiques d'avant 1980, très appréciés")
    df_classics = df[(df['release_date'].dt.year < 1980) & (df['vote_average'] > 7.5)]
    afficher_film_aleatoire(df_classics, "Voir un classique au hasard", "btn_classic")



with tab3:
    st.write("Films d'art & essai (Documentary, Drama, etc.)")
    keywords = ['Documentary', 'Drama', 'Romance', 'History', 'Biography']
    mask_arts = df['genres'].str.contains('|'.join(keywords), case=False, na=False)
    df_arts = df[mask_arts & (df['vote_average'] > 7)]
    afficher_film_aleatoire(df_arts, "Voir un film d’art & essai au hasard", "btn_art")



with tab4:
    choix_langue = st.selectbox("Choisissez une langue", df['original_language'].unique())
    df_langue = df[df['original_language'] == choix_langue]
    if df_langue.empty:
        st.warning("Aucun film dans cette langue.")
    else:
        afficher_film_aleatoire(df_langue, f"Voir un film en {choix_langue.upper()} au hasard", "btn_langue")

@st.cache_data
def load_all_models():
    features_df = pd.read_csv("models/features_df.csv")
    with open("models/knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
    return features_df, knn_model

features_df, knn_model = load_all_models()

def recommend_movie(movie_id, n_recommendations=5):
    try:
        idx = df[df["id"] == movie_id].index[0]
    except IndexError:
        st.error("Film ID non trouvé.")
        return []

    distances, indices = knn_model.kneighbors(
        [features_df.iloc[idx].values],
        n_neighbors=n_recommendations + 1
    )
    return indices.flatten()[1:]


with st.expander("Rechercher un film manuellement"):
    titre_input = st.text_input("Entrez un mot clé du titre :", key="search_main")

    if titre_input:
        resultats = df[df['original_title'].str.contains(titre_input, case=False, na=False)].sort_values("popularity", ascending=False).head(10)
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
        film_row = get_enriched_film(film_row)

    afficher_film(film_row)

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
            rec_film = get_enriched_film(rec_film)
            with cols[i]:
                with st.spinner(f"Chargement : {rec_film['title']}"):
                    st.image(rec_film['poster_url'] if pd.notna(rec_film['poster_url']) else DEFAULT_POSTER, width=140)
                    st.markdown(f"**{rec_film['title']}**", unsafe_allow_html=True)
                    st.caption(f"{rec_film['release_date'].year} | {rec_film['vote_average']}/10")
                    if st.button(f"{rec_film['title']}", key=f"img_{rec_film['id']}"):
                        st.session_state.current_movie_id = rec_film['id']
                        st.rerun()


st.markdown("---")
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("logo_wild.png", width=50)
with col2:
    st.markdown("Cette application a été réalisée par **F.C Data, Ahcene K, Kamel T & Majed S**, Wild Code School 2025.")