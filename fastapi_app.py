from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from typing import List

app = FastAPI()

# Load data and model
movies_df = pd.read_csv('notebooks/movies_df.csv')
with open('notebooks/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

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

# To run: uvicorn fastapi_app:app --reload
