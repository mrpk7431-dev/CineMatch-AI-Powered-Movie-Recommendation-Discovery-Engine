import joblib
import os
import difflib

class RecommendationService:
    def __init__(self):
        # Load artifacts
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.movie_data = joblib.load(os.path.join(base_path, 'models', 'movie_data.pkl'))
        self.similarity_matrix = joblib.load(os.path.join(base_path, 'models', 'similarity_matrix.pkl'))
        
        # Create a list of all movie titles for fast searching
        self.all_titles = self.movie_data['Movie_Title'].tolist()
        
    def get_movie_details(self, title=None, movie_id=None):
        if movie_id is not None:
            movie = self.movie_data[self.movie_data['Movie_ID'] == int(movie_id)]
        elif title:
            movie = self.movie_data[self.movie_data['Movie_Title'].str.lower() == title.lower()]
        else:
            return None
            
        if movie.empty:
            return None
            
        # Convert first match to dict
        return movie.iloc[0].to_dict()

    def search_movies(self, query, limit=5):
        # Exact match or substring
        matches = [t for t in self.all_titles if query.lower() in str(t).lower()]
        # If not enough substring matches, try difflib
        if len(matches) < limit:
            close_matches = difflib.get_close_matches(query, [str(t) for t in self.all_titles], n=limit, cutoff=0.5)
            for m in close_matches:
                if m not in matches:
                    matches.append(m)
        return matches[:limit]

    def recommend_movies(self, movie_title, top_n=10):
        # Find close match for the movie title just in case
        close_matches = difflib.get_close_matches(movie_title, [str(t) for t in self.all_titles])
        if not close_matches:
            return {"error": "Movie not found"}
            
        matched_title = close_matches[0]
        
        # Get index
        idx = self.movie_data[self.movie_data['Movie_Title'] == matched_title].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n (skip the first one because it's the movie itself)
        sim_scores = sim_scores[1:top_n+1]
        
        recommendations = []
        for i, score in sim_scores:
            row = self.movie_data.iloc[i]
            recommendations.append({
                "id": row['Movie_ID'],
                "title": row['Movie_Title'],
                "poster_path": "", # We don't have this in the CSV, but we can return it empty
                "year": row['Movie_Release_Date'][:4] if isinstance(row['Movie_Release_Date'], str) else "N/A",
                "rating": row['Movie_Vote'],
                "score": round(score * 100),
                "reasons": self._generate_reasons(self.movie_data.iloc[idx], row)
            })
            
        return {
            "movie": self.get_movie_details(title=matched_title),
            "recommendations": recommendations
        }
        
    def _generate_reasons(self, original_movie, recommended_movie):
        reasons = []
        # Check genres
        orig_genres = set(str(original_movie['Movie_Genre']).lower().split())
        rec_genres = set(str(recommended_movie['Movie_Genre']).lower().split())
        common_genres = orig_genres.intersection(rec_genres)
        if common_genres:
            reasons.append(f"Similar genres")
            
        # Check cast
        orig_cast = set(str(original_movie['Movie_Cast']).lower().split())
        rec_cast = set(str(recommended_movie['Movie_Cast']).lower().split())
        if orig_cast.intersection(rec_cast):
            reasons.append(f"Similar cast members")
            
        # Check director
        if str(original_movie['Movie_Director']).lower() == str(recommended_movie['Movie_Director']).lower() and str(original_movie['Movie_Director']) != 'nan':
            reasons.append(f"Same director")
            
        # Default fallback
        if not reasons:
            reasons.append("Similar overall theme")
            
        return reasons
        
    def get_movies_by_mood(self, mood, limit=10):
        # A simple keyword mapping for moods
        mood_mapping = {
            'happy': ['comedy', 'family', 'animation', 'funny'],
            'romantic': ['romance', 'love'],
            'exciting': ['action', 'thriller', 'adventure'],
            'mind-bending': ['sci-fi', 'mystery', 'puzzle', 'time travel', 'science fiction'],
            'emotional': ['drama', 'sad', 'touching'],
            'scary': ['horror', 'creepy', 'scary', 'ghost'],
            'adventurous': ['adventure', 'journey', 'space'],
            'relaxing': ['nature', 'documentary', 'slow']
        }
        
        keywords = mood_mapping.get(mood.lower(), [])
        
        # Score movies based on how many keywords they match in their combined features
        def score_movie(row):
            text = str(row['combined_features']).lower()
            return sum(1 for kw in keywords if kw in text)
            
        scores = self.movie_data.apply(score_movie, axis=1)
        top_indices = scores.nlargest(limit).index
        
        results = []
        for i in top_indices:
            row = self.movie_data.iloc[i]
            results.append({
                "id": row['Movie_ID'],
                "title": row['Movie_Title'],
                "year": row['Movie_Release_Date'][:4] if isinstance(row['Movie_Release_Date'], str) else "N/A",
                "rating": row['Movie_Vote'],
                "score": "Mood Match"
            })
        return results

    def get_explore_movies(self, category='popular', limit=20):
        df = self.movie_data.copy()
        if category == 'popular':
            df['Movie_Popularity'] = pd.to_numeric(df['Movie_Popularity'], errors='coerce')
            df = df.sort_values(by='Movie_Popularity', ascending=False)
        elif category == 'top_rated':
            df['Movie_Vote'] = pd.to_numeric(df['Movie_Vote'], errors='coerce')
            df = df.sort_values(by='Movie_Vote', ascending=False)
            
        results = []
        for _, row in df.head(limit).iterrows():
            results.append({
                "id": row['Movie_ID'],
                "title": row['Movie_Title'],
                "year": row['Movie_Release_Date'][:4] if isinstance(row['Movie_Release_Date'], str) else "N/A",
                "rating": row['Movie_Vote']
            })
        return results
