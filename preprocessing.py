
#importation des bibliothèques
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# lecture du CSV clean
df = pd.read_csv("data_clean.csv", parse_dates=['release_date'])

#prendre que l'année de sortie du film
df['release_year'] = df['release_date'].dt.year.astype(int)

# désocier les genres 
df['genres_list'] = df['genres'].apply(lambda x: x.split(' ') if pd.notnull(x) else [])

# multilabel binarizer sur genres_list
mlb_genres = MultiLabelBinarizer()
genres_encoded = mlb_genres.fit_transform(df['genres_list'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_, index=df.index)

# one hot encoded sur original_language
ohe_lang = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
lang_encoded = ohe_lang.fit_transform(df[['original_language']])
lang_df = pd.DataFrame(lang_encoded, columns=ohe_lang.get_feature_names_out(), index=df.index)


#SAGA

# On garde les saga_name_clean fréquentes (>= 2 apparitions)                                        # à supprimer ?
vc = df['saga_name_clean'].value_counts()                                                                 # à supprimer ?
sagas_valides = vc[vc >= 10].index.tolist()                                                         # à supprimer ?

# On remplace les autres par 'Other'
df['saga_name_clean'] = df['saga_name_clean'].apply(
    lambda x: x if x in sagas_valides else 'Other'
)                                                                                                   # à supprimer ?

ohe_saga_cleaned = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
saga_encoded = ohe_saga_cleaned.fit_transform(df[['saga_name_clean']])
saga_df = pd.DataFrame(saga_encoded, columns=ohe_saga_cleaned.get_feature_names_out(), index=df.index)

# appliquer un logarythmique +1 afin de scaler le bruit de vote_count
df['vote_count_log'] = np.log1p(df['vote_count'])


# minmaxscaler : mettre à éhelle entre zéro et 1 des colonnes :
num_cols = ['vote_average', 'vote_count_log', 'runtime', 'popularity', 'release_year']
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols, index=df.index)


# concatener tout en features : 
#prioriser les films de la meme sage, multiplier le poids de saga dans le modèle : 
saga_df_weighted = saga_df * 5

features_df = pd.concat([genres_df, lang_df, saga_df_weighted, df_scaled], axis=1)


#fit le modèle
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(features_df.values)

# sauvgarder les résultats pour réutilisation et alléger la ram de l'appli
with open('features_df.pkl', 'wb') as f:
    pickle.dump(features_df, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('mlb_genres.pkl', 'wb') as f:
    pickle.dump(mlb_genres, f)

with open('ohe_lang.pkl', 'wb') as f:
    pickle.dump(ohe_lang, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('ohe_saga_cleaned.pkl', 'wb') as f:
    pickle.dump(saga_df, f)


print(f'modèle fonctionnel et résultats pickled avec succés')
