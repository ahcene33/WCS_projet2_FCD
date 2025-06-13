# preprocessing.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import pickle

# Chargement des données
data_tmdb = pd.read_csv("data_tmdb_clean.csv")
films = data_tmdb.copy()
films['text_features'] = films['text_features'].fillna('').astype(str)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=6000)
tfidf_matrix = tfidf.fit_transform(films['text_features'])

# SVD
svd = TruncatedSVD(n_components=300, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tfidf_reduced)

# Sauvegarde pickle
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)

with open('tfidf_reduced.pkl', 'wb') as f:
    pickle.dump(tfidf_reduced, f)

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("pickle sauvgardé avec succés !")
