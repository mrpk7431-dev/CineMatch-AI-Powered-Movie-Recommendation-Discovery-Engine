import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import load_data, clean_data
from feature_engineering import create_combined_features

def main():
    print("Loading dataset...")
    df = load_data()
    
    print("Cleaning data...")
    df_cleaned = clean_data(df)
    
    print("Creating combined features...")
    df_features = create_combined_features(df_cleaned)
    
    print("Training TF-IDF model...")
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    feature_vectors = vectorizer.fit_transform(df_features['combined_features'])
    
    print("Calculating similarity...")
    # Calculate cosine similarity
    similarity = cosine_similarity(feature_vectors)
    
    print("Saving model...")
    # Save the vectorizer, similarity matrix, and processed dataset
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    joblib.dump(vectorizer, 'models/movie_vectorizer.pkl')
    joblib.dump(similarity, 'models/similarity_matrix.pkl')
    df_features.to_pickle('models/movie_data.pkl')
    
    # Save clean dataset
    df_features.to_csv('data/processed/movies_cleaned.csv', index=False)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
