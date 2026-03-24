import pandas as pd
import ast
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def extract_names(obj, max_items=None):
    try:
        items = ast.literal_eval(obj)
        names = [i['name'].replace(" ", "") for i in items]
        if max_items:
            names = names[:max_items]
        return " ".join(names)
    except (ValueError, SyntaxError, TypeError):
        return ""

def extract_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return i['name'].replace(" ", "")
        return ""
    except (ValueError, SyntaxError, TypeError):
        return ""

def train():
    print("Loading datasets...")
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    
    print("Extracting features (Genres, Keywords, Cast, Director)...")
    movies['genres'] = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    
     
    movies['cast'] = movies['cast'].apply(lambda x: extract_names(x, max_items=3))
    movies['director'] = movies['crew'].apply(extract_director)
    
    print("Building content tags...")
  
    movies['tags'] = (
        movies['overview'].fillna('') + ' ' + 
        movies['genres'] + ' ' + 
        movies['keywords'] + ' ' + 
        movies['cast'] + ' ' + 
        movies['director']
    )
    
    movies_clean = movies[['title', 'tags']].reset_index(drop=True)

    print("Vectorizing tags using TF-IDF...")
   
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(movies_clean['tags']) # Keeps as sparse matrix

    print("Training KNN model...")
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='auto')
    knn.fit(vectors)

    print("Saving assets...")
    os.makedirs('model', exist_ok=True)
    
    with open('model/knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
        
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
   
    movies_clean.to_pickle('model/movies.pkl')
    movies_clean.to_csv('model/movies.csv', index=False)

    print("✅ Training complete! Models saved to model/")

if __name__ == '__main__':
    train()
