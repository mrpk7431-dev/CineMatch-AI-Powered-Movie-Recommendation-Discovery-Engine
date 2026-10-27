import pandas as pd

df = pd.read_csv("data/raw/movies.csv")
print("Columns:", df.columns.tolist())
print("\nFirst row:\n", df.iloc[0])
