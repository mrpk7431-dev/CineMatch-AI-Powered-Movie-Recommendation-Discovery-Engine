import joblib
import pandas as pd
import numpy as np
import os
import random

def evaluate_model():
    print("Evaluating Model Recommendations...")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if models exist
    try:
        movie_data = joblib.load(os.path.join(base_path, 'models', 'movie_data.pkl'))
        similarity_matrix = joblib.load(os.path.join(base_path, 'models', 'similarity_matrix.pkl'))
    except FileNotFoundError:
        print("Models not found. Please run train.py first.")
        return
        
    print(f"Loaded {len(movie_data)} movies for evaluation.")
    
    # 1. Random Sample Sanity Check
    print("\n--- SANITY CHECK ---")
    random_idx = random.randint(0, len(movie_data)-1)
    movie = movie_data.iloc[random_idx]
    
    print(f"Test Movie: {movie['Movie_Title']}")
    print(f"Genres: {movie['Movie_Genre']}")
    
    # Get top 5 recommendations
    sim_scores = list(enumerate(similarity_matrix[random_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    
    print("\nTop 5 Recommendations:")
    for i, score in sim_scores:
        rec_movie = movie_data.iloc[i]
        print(f"- {rec_movie['Movie_Title']} (Score: {score:.3f}) - Genres: {rec_movie['Movie_Genre']}")
        
    # Note: Since this is an unsupervised content-based recommender, 
    # we don't have ground truth labels (like user ratings) to calculate standard metrics like RMSE or NDCG.
    # Evaluation for content-based models is mostly qualitative and based on feature overlap.
    
    print("\nEvaluation Note: As this is a content-based recommendation system without user interaction logs,")
    print("standard metrics (Precision@K, Recall, RMSE) cannot be calculated against ground truth.")
    print("The system successfully calculates mathematical cosine similarity between feature vectors.")

if __name__ == "__main__":
    evaluate_model()
