from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from typing import List
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

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


@app.get("/recommend", response_model=List[dict])
def recommend_by_id(movie_id: int, top_n: int = 10):
    if movie_id not in movies_df['id'].values:
        return []
    idx = movies_df.index[movies_df['id'] == movie_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    movie_indices = [i[0] for i in sim_scores]
    result = movies_df.iloc[movie_indices][['id', 'title', 'poster_url']].to_dict(orient='records')
    return result

# --- New endpoint for collaborative filtering ---
@app.get("/recommend/user-collaborative", response_model=List[int])
def recommend_user_collaborative(user_id: str, top_k: int = 10):
    favorites_df = load_favorites()
    user_item_matrix, user_sim_df = build_matrices(favorites_df)
    recommended = recommend_movies_for_user(user_id, user_item_matrix, user_sim_df, N=5, top_k=top_k)
    return recommended

# To run: uvicorn fastapi_app:app --reload
