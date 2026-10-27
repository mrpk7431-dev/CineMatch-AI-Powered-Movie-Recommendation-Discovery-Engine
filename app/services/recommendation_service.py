import joblib
import os
import difflib
import re
from sklearn.metrics.pairwise import cosine_similarity
from app.services.movie_api_service import MovieAPIService

class RecommendationService:
    def __init__(self):
        # Load ML artifacts
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        try:
            self.movie_data = joblib.load(os.path.join(base_path, 'models', 'movie_data.pkl'))
            self.similarity_matrix = joblib.load(os.path.join(base_path, 'models', 'similarity_matrix.pkl'))
            self.all_titles = self.movie_data['Movie_Title'].tolist()
            
            # Load vectorizer for online/new movie recommendation
            vectorizer_path = os.path.join(base_path, 'models', 'movie_vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
                self.feature_vectors = self.vectorizer.transform(self.movie_data['combined_features'])
            else:
                self.vectorizer = None
                self.feature_vectors = None
                
            self.is_ready = True
        except Exception as e:
            print(f"ML Model Error: {e}")
            self.is_ready = False
            
        self.api_service = MovieAPIService()
        
    def _extract_year(self, date_val):
        if not date_val or str(date_val).strip() in ("N/A", "", "nan", "None"):
            return "N/A"
        m = re.search(r'\b(19\d\d|20\d\d)\b', str(date_val))
        if m:
            return m.group(1)
        parts = str(date_val).strip().split('-')
        if len(parts) == 3 and len(parts[-1]) == 4:
            return parts[-1]
        return str(date_val)[:4]
        
    def recommend_movies(self, movie_title, top_n=10):
        if not self.is_ready:
            return {"error": "Recommendation model not ready"}
            
        # 1. First attempt to match title to dataset
        close_matches = difflib.get_close_matches(movie_title, [str(t) for t in self.all_titles], n=1, cutoff=0.6)
        
        if close_matches:
            matched_title = close_matches[0]
            idx = self.movie_data[self.movie_data['Movie_Title'] == matched_title].index[0]
            
            # Get similarity scores from precomputed matrix
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            
            recommendations = []
            original_movie_row = self.movie_data.iloc[idx]
            
            for i, score in sim_scores:
                row = self.movie_data.iloc[i]
                title = row['Movie_Title']
                real_details = self.api_service.get_movie_details(title=title)
                
                if real_details:
                    real_details["score"] = max(50, min(99, round(score * 100)))
                    real_details["reasons"] = self._generate_reasons(original_movie_row, row)
                    recommendations.append(real_details)
                else:
                    recommendations.append({
                        "movie_id": row['Movie_ID'],
                        "title": title,
                        "poster_url": None,
                        "release_year": self._extract_year(row.get('Movie_Release_Date')),
                        "rating": row['Movie_Vote'],
                        "overview": row['Movie_Overview'],
                        "genre": str(row['Movie_Genre']),
                        "score": max(50, min(99, round(score * 100))),
                        "reasons": self._generate_reasons(original_movie_row, row)
                    })
                    
            real_original = self.api_service.get_movie_details(title=matched_title) or {
                "title": matched_title,
                "poster_url": None
            }
            
            return {
                "movie": real_original,
                "recommendations": recommendations
            }

        # 2. If not in dataset, use online API + trained vectorizer!
        online_details = self.api_service.get_movie_details(title=movie_title)
        if not online_details:
            return {"error": f"Movie '{movie_title}' not found in training dataset or online."}

        if self.vectorizer is not None and self.feature_vectors is not None:
            # Combine features from online details
            combined_text = f"{online_details.get('genre', '')} {online_details.get('overview', '')} {online_details.get('cast', '')} {online_details.get('director', '')}"
            new_vec = self.vectorizer.transform([combined_text])
            scores = cosine_similarity(new_vec, self.feature_vectors)[0]
            top_indices = scores.argsort()[::-1][:top_n]
            
            # Simulated original row for reason comparison
            orig_pseudo_row = {
                'Movie_Genre': online_details.get('genre', ''),
                'Movie_Cast': online_details.get('cast', ''),
                'Movie_Director': online_details.get('director', '')
            }
            
            recommendations = []
            for i in top_indices:
                score = scores[i]
                row = self.movie_data.iloc[i]
                title = row['Movie_Title']
                real_details = self.api_service.get_movie_details(title=title)
                
                calculated_score = max(50, min(98, round(float(score) * 100)))
                if real_details:
                    real_details["score"] = calculated_score
                    real_details["reasons"] = self._generate_reasons(orig_pseudo_row, row)
                    recommendations.append(real_details)
                else:
                    recommendations.append({
                        "movie_id": row['Movie_ID'],
                        "title": title,
                        "poster_url": None,
                        "release_year": self._extract_year(row.get('Movie_Release_Date')),
                        "rating": row['Movie_Vote'],
                        "overview": row['Movie_Overview'],
                        "genre": str(row['Movie_Genre']),
                        "score": calculated_score,
                        "reasons": self._generate_reasons(orig_pseudo_row, row)
                    })
                    
            return {
                "movie": online_details,
                "recommendations": recommendations
            }
            
        return {"error": "Vector model unavailable for online recommendations."}
        
    def _generate_reasons(self, original_movie, recommended_movie):
        reasons = []
        # Check genres
        orig_genres = set(str(original_movie.get('Movie_Genre', '')).lower().replace(',', ' ').split())
        rec_genres = set(str(recommended_movie.get('Movie_Genre', '')).lower().replace(',', ' ').split())
        common_genres = orig_genres.intersection(rec_genres)
        if common_genres:
            genres_list = [g.capitalize() for g in list(common_genres)[:2] if len(g) > 2]
            if genres_list:
                reasons.append(f"Matching genres: {', '.join(genres_list)}")
            else:
                reasons.append("Similar genre & style")
            
        # Check cast
        orig_cast = set(str(original_movie.get('Movie_Cast', '')).lower().replace(',', ' ').split())
        rec_cast = set(str(recommended_movie.get('Movie_Cast', '')).lower().replace(',', ' ').split())
        common_cast = [c for c in orig_cast.intersection(rec_cast) if len(c) > 3]
        if common_cast:
            reasons.append("Overlapping cast or crew")
            
        # Check director
        orig_dir = str(original_movie.get('Movie_Director', '')).lower()
        rec_dir = str(recommended_movie.get('Movie_Director', '')).lower()
        if orig_dir and rec_dir and orig_dir == rec_dir and orig_dir not in ('nan', '', 'none', 'n/a'):
            reasons.append("Same director vision")
            
        if not reasons:
            reasons.append("Deep narrative & thematic affinity")
            
        return reasons
