from flask import Flask, render_template, request, jsonify
from app.services.recommendation_service import RecommendationService
from app.services.movie_api_service import MovieAPIService
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
recommender = RecommendationService()
movie_api = MovieAPIService()

AVAILABLE_YEARS = [
    {"value": "2026", "label": "2026 (Current)", "short_label": "2026", "is_current": True},
    {"value": "2025", "label": "2025", "short_label": "2025", "is_current": False},
    {"value": "2024", "label": "2024", "short_label": "2024", "is_current": False},
    {"value": "2023", "label": "2023", "short_label": "2023", "is_current": False},
    {"value": "2022", "label": "2022", "short_label": "2022", "is_current": False},
    {"value": "2021", "label": "2021", "short_label": "2021", "is_current": False},
    {"value": "2020", "label": "2020", "short_label": "2020", "is_current": False},
    {"value": "2019", "label": "2019", "short_label": "2019", "is_current": False},
    {"value": "2018", "label": "2018", "short_label": "2018", "is_current": False},
    {"value": "2017", "label": "2017", "short_label": "2017", "is_current": False},
    {"value": "2016", "label": "2016", "short_label": "2016", "is_current": False},
    {"value": "2015", "label": "2015", "short_label": "2015", "is_current": False},
    {"value": "2014", "label": "2014", "short_label": "2014", "is_current": False},
    {"value": "2013", "label": "2013", "short_label": "2013", "is_current": False},
    {"value": "2012", "label": "2012", "short_label": "2012", "is_current": False},
    {"value": "2011", "label": "2011", "short_label": "2011", "is_current": False},
    {"value": "2010", "label": "2010", "short_label": "2010", "is_current": False},
    {"value": "2000s", "label": "2000s", "short_label": "2000s", "is_current": False},
    {"value": "1990s", "label": "1990s", "short_label": "1990s", "is_current": False},
    {"value": "1980s", "label": "1980s", "short_label": "1980s", "is_current": False},
    {"value": "classic", "label": "Classic", "short_label": "Classic", "is_current": False},
]

@app.context_processor
def inject_globals():
    return dict(
        has_api_key=bool(os.getenv("TMDB_API_KEY")),
        available_years=AVAILABLE_YEARS
    )

@app.route('/')
def home():
    popular = movie_api.get_popular_movies(limit=20)
    return render_template('index.html', popular=popular, available_years=AVAILABLE_YEARS)

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return render_template('results.html', query=query, results=[])
    
    movies = movie_api.search_movies(query)
    return render_template('results.html', query=query, results=movies, available_years=AVAILABLE_YEARS)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    movies = movie_api.search_movies(query)[:8]
    return jsonify([m['title'] for m in movies if m])

@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        movie = request.form.get('movie')
    else:
        movie = request.args.get('movie')
        
    if not movie:
        return render_template('index.html', error="Please enter a movie title.", available_years=AVAILABLE_YEARS)
        
    result = recommender.recommend_movies(movie, top_n=10)
    
    if "error" in result:
        return render_template('index.html', error=result["error"], available_years=AVAILABLE_YEARS)
        
    return render_template('results.html', 
                          query=movie, 
                          original_movie=result['movie'],
                          recommendations=result['recommendations'],
                          is_recommendation=True,
                          available_years=AVAILABLE_YEARS)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    movie = movie_api.get_movie_details(movie_id=movie_id)
    if not movie:
        return render_template('index.html', error="Movie not found.", available_years=AVAILABLE_YEARS)
        
    # Get recommendations based on this movie using trained ML vectorizer
    result = recommender.recommend_movies(movie['title'], top_n=6)
    similar_movies = result.get('recommendations', []) if "error" not in result else []
        
    return render_template('movie.html', movie=movie, similar_movies=similar_movies, available_years=AVAILABLE_YEARS)

@app.route('/explore')
def explore():
    genre = request.args.get('genre', '').strip()
    year = request.args.get('year', '').strip()
    language = request.args.get('language', '').strip()
    
    # If no filters provided but accessed via /explore directly
    if not genre and not year and not language:
        category = request.args.get('category', 'popular')
        if category == 'popular':
            movies = movie_api.get_popular_movies(limit=20)
            query = "Popular Movies"
        else:
            movies = movie_api.get_trending_movies(limit=20)
            query = "Trending Movies"
    else:
        movies = movie_api.filter_movies(genre=genre, year=year, language=language, limit=20)
        
        # Build friendly title
        year_label = year
        for y in AVAILABLE_YEARS:
            if y["value"] == year:
                year_label = y["label"]
                break
                
        if year and genre:
            query = f"Top 20 {genre} Movies of {year_label}"
        elif year:
            query = f"Top 20 Movies of {year_label}"
        elif genre:
            query = f"Top 20 {genre} Movies"
        else:
            query = "Top 20 Filtered Movies"
        
    return render_template('results.html', 
                          query=query, 
                          results=movies, 
                          is_explore=True,
                          selected_year=year,
                          selected_genre=genre,
                          available_years=AVAILABLE_YEARS)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
