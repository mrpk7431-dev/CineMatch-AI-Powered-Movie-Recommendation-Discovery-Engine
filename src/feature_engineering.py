import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ''
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def create_combined_features(df):
    print("Creating combined features...")
    df_features = df.copy()
    
    # Text columns to combine
    cols_to_combine = ['Movie_Genre', 'Movie_Overview', 'Movie_Keywords', 'Movie_Cast', 'Movie_Director']
    
    # Clean text in each column
    for col in cols_to_combine:
        if col in df_features.columns:
            df_features[col] = df_features[col].apply(clean_text)
    
    # Combine features
    def combine(row):
        features = []
        for col in cols_to_combine:
            if col in row and pd.notna(row[col]):
                features.append(str(row[col]))
        return ' '.join(features)
        
    df_features['combined_features'] = df_features.apply(combine, axis=1)
    
    return df_features
