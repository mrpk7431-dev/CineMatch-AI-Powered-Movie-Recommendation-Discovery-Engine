import urllib.request
import os

url = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv"
output_path = os.path.join("data", "raw", "movies.csv")

print("Downloading dataset...")
urllib.request.urlretrieve(url, output_path)
print(f"Dataset downloaded and saved to {output_path}")
