import pandas as pd

def load_data(filepath="data/raw/movies.csv"):
    return pd.read_csv(filepath)

def clean_data(df):
    print("Cleaning data...")
    # Select only the features we need
    features = ['Movie_ID', 'Movie_Title', 'Movie_Genre', 'Movie_Overview', 'Movie_Keywords', 'Movie_Cast', 'Movie_Director', 'Movie_Vote', 'Movie_Release_Date']
    
    # Handle cases where some columns might be missing just in case
    existing_features = [f for f in features if f in df.columns]
    
    cleaned_df = df[existing_features].copy()
    
    # Drop rows where Movie_Title is null
    cleaned_df.dropna(subset=['Movie_Title'], inplace=True)
    
    # Fill NaN values with empty strings for text columns
    text_cols = ['Movie_Genre', 'Movie_Overview', 'Movie_Keywords', 'Movie_Cast', 'Movie_Director']
    for col in text_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna('')
    
    # Drop duplicate movies if any based on title
    cleaned_df.drop_duplicates(subset=['Movie_Title'], inplace=True)
    
    # Reset index
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

if __name__ == "__main__":
    df = load_data()
    cleaned = clean_data(df)
    cleaned.to_csv("data/processed/movies_cleaned.csv", index=False)
    print("Data cleaned and saved.")
