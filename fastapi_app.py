from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from typing import List
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = FastAPI()

movies_df = pd.read_csv('notebooks/movies_df.csv')
with open('notebooks/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Collaborative Filtering Setup ---
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def load_favorites():
    response = supabase.table('favorites').select('*').execute()
    favorites_data = response.data
    return pd.DataFrame(favorites_data)

def build_matrices(favorites_df):
    user_item_matrix = favorites_df.pivot_table(index='user_id', columns='movie_id', values='id', aggfunc='count', fill_value=0)
    user_item_matrix[user_item_matrix > 0] = 1
    user_sim_matrix = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_item_matrix, user_sim_df

def recommend_movies_for_user(user_id, user_item_matrix, user_sim_df, N=5, top_k=10):
    if user_id not in user_item_matrix.index:
        return []
    similar_users = user_sim_df.loc[user_id].drop(user_id).sort_values(ascending=False).head(N).index.tolist()
    movies_similar_users_like = user_item_matrix.loc[similar_users].sum(axis=0)
    movies_user_liked = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 1].index)
    recommend_movies = movies_similar_users_like.drop(labels=movies_user_liked)
    recommend_movies = recommend_movies[recommend_movies > 0].sort_values(ascending=False)
    return recommend_movies.index.tolist()[:top_k]

def preprocess_text(text):
    """Preprocess text for TF-IDF"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    except:
        # Fallback if NLTK data not available
        words = text.split()
        return ' '.join([word for word in words if len(word) > 2])

def rebuild_tfidf_matrix(movies_df):
    """Rebuild TF-IDF matrix from movies dataframe"""
    print("Rebuilding TF-IDF matrix...")
    
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    
    # Combine text features for content-based filtering
    movies_df['combined_features'] = (
        movies_df['title'].fillna('') + ' ' +
        movies_df['overview'].fillna('') + ' ' +
        movies_df['genre'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('')
    )
    
    # Preprocess combined features
    movies_df['processed_features'] = movies_df['combined_features'].apply(preprocess_text)
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    tfidf_matrix = tfidf.fit_transform(movies_df['processed_features'])
    
    # Save the updated TF-IDF matrix
    with open('notebooks/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    print(f"TF-IDF matrix rebuilt with shape: {tfidf_matrix.shape}")
    return tfidf_matrix


@app.get("/recommend", response_model=List[dict])
def recommend_by_id(movie_id: int, top_n: int = 10):
    if movie_id not in movies_df['id'].values:
        return []
    idx = movies_df.index[movies_df['id'] == movie_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    movie_indices = [i[0] for i in sim_scores]
    result = movies_df.iloc[movie_indices][['id', 'title', 'poster_path']].to_dict(orient='records')
    return result

# --- New endpoint for collaborative filtering ---
@app.get("/recommend/user-collaborative", response_model=List[int])
def recommend_user_collaborative(user_id: str, top_k: int = 10):
    favorites_df = load_favorites()
    user_item_matrix, user_sim_df = build_matrices(favorites_df)
    recommended = recommend_movies_for_user(user_id, user_item_matrix, user_sim_df, N=5, top_k=top_k)
    return recommended

# --- Endpoint to update movie data ---
@app.post("/update-movies")
def update_movies():
    """
    Endpoint to trigger the fetch_new_movies function on Supabase
    and update local movie data
    """
    try:
        # Call the Supabase Edge Function to fetch new movies
        response = supabase.functions.invoke('fetch_new_movies', invoke_options={'body': {}})
        import json
        if isinstance(response, bytes):
            response_json = json.loads(response.decode("utf-8"))
        else:
            response_json = response

        if response_json.get('error') is None:
            # Reload the movies data after successful fetch
            global movies_df, tfidf_matrix, cosine_sim
            
            # Load updated movies data from Supabase with pagination
            print("Loading updated movies from Supabase...")
            all_movies = []
            page_size = 1000
            start = 0
            
            while True:
                movies_response = supabase.table('movies').select('*').range(start, start + page_size - 1).execute()
                if not movies_response.data:
                    break
                    
                all_movies.extend(movies_response.data)
                print(f"Loaded {len(movies_response.data)} movies (total: {len(all_movies)})")
                
                # If we got less than page_size records, we've reached the end
                if len(movies_response.data) < page_size:
                    break
                    
                start += page_size
            
            if all_movies:
                movies_df = pd.DataFrame(all_movies)
                
                # Save updated movies to CSV for backup
                movies_df.to_csv('notebooks/movies_df.csv', index=False)
                
                # Rebuild TF-IDF matrix with new data
                tfidf_matrix = rebuild_tfidf_matrix(movies_df)
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                return {
                    "success": True,
                    "message": "Movies data and TF-IDF matrix updated successfully",
                    "total_movies": len(movies_df),
                    "tfidf_shape": list(tfidf_matrix.shape),
                    "function_response": response_json
                }
            else:
                # Fallback to CSV reload
                movies_df = pd.read_csv('notebooks/movies_df.csv')
                with open('notebooks/tfidf_matrix.pkl', 'rb') as f:
                    tfidf_matrix = pickle.load(f)
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                return {
                    "success": True,
                    "message": "Movies data updated from CSV (Supabase query failed)",
                    "total_movies": len(movies_df),
                    "function_response": response_json
                }
        else:
            return {
                "success": False,
                "message": "Failed to fetch new movies",
                "error": response_json.get('error')
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating movies: {str(e)}")

# To run: uvicorn fastapi_app:app --reload
