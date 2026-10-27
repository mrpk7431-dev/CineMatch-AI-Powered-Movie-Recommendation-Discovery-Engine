import requests
import joblib
import pandas as pd
import numpy as np
import os
import re
from app.services.cache_service import CacheService
import concurrent.futures

RECENT_YEAR_MOVIES = {
    '2026': [
        "Avengers: Doomsday", "The Batman: Part II", "Avatar 3", "Project Hail Mary",
        "Spider-Man 4", "Supergirl: Woman of Tomorrow", "The Mandalorian & Grogu", "Dune: Messiah",
        "Toy Story 5", "Tron: Ares", "Fast X: Part 2", "Scream 7", "Shrek 5",
        "Masters of the Universe", "Weapons", "Moana", "Blade", "Wicked: Part Two",
        "The Cat in the Hat", "Ice Age 6"
    ],
    '2025': [
        "Superman", "The Fantastic Four: First Steps", "Avatar: Fire and Ash", "Jurassic World Rebirth",
        "Mickey 17", "Captain America: Brave New World", "Mission: Impossible - The Final Reckoning",
        "Thunderbolts*", "Sinners", "F1", "28 Years Later", "How to Train Your Dragon", "Snow White",
        "Ballerina", "Zootopia 2", "Predator: Badlands", "Karate Kid: Legends", "Michael",
        "Five Nights at Freddy's 2", "The Amateur"
    ],
    '2024': [
        "Dune: Part Two", "Deadpool & Wolverine", "Inside Out 2", "Gladiator II", "The Wild Robot",
        "Wicked", "Alien: Romulus", "Furiosa: A Mad Max Saga", "Challengers", "Civil War",
        "Beetlejuice Beetlejuice", "Twisters", "Kingdom of the Planet of the Apes", "Longlegs",
        "The Fall Guy", "Godzilla x Kong: The New Empire", "A Quiet Place: Day One", "Bad Boys: Ride or Die",
        "Despicable Me 4", "Kung Fu Panda 4"
    ],
    '2023': [
        "Oppenheimer", "Spider-Man: Across the Spider-Verse", "Barbie", "Guardians of the Galaxy Vol. 3",
        "Past Lives", "Poor Things", "Killers of the Flower Moon", "Anatomy of a Fall", "The Holdovers",
        "The Zone of Interest", "Godzilla Minus One", "John Wick: Chapter 4",
        "Mission: Impossible - Dead Reckoning", "The Boy and the Heron", "The Iron Claw", "Talk to Me",
        "Wonka", "Monster", "Asteroid City", "Society of the Snow"
    ],
    '2022': [
        "Everything Everywhere All at Once", "Top Gun: Maverick", "The Batman", "Avatar: The Way of Water",
        "The Banshees of Inisherin", "Tár", "Guillermo del Toro's Pinocchio", "Puss in Boots: The Last Wish",
        "Aftersun", "The Fabelmans", "Glass Onion", "RRR", "Babylon", "All Quiet on the Western Front",
        "The Whale", "Black Panther: Wakanda Forever", "Elvis", "Nope", "Decision to Leave", "Triangle of Sadness"
    ],
    '2021': [
        "Dune", "Spider-Man: No Way Home", "CODA", "The Power of the Dog", "Drive My Car",
        "Licorice Pizza", "West Side Story", "The French Dispatch", "Belfast",
        "The Worst Person in the World", "No Time to Die", "tick, tick... BOOM!", "Encanto",
        "Nightmare Alley", "Don't Look Up", "The Green Knight", "The Suicide Squad", "Luca",
        "The Mitchells vs. the Machines", "Spencer"
    ],
    '2020': [
        "Soul", "Another Round", "Sound of Metal", "The Father", "Nomadland", "Minari",
        "Promising Young Woman", "Tenet", "The Trial of the Chicago 7", "Palm Springs",
        "Da 5 Bloods", "Wolfwalkers", "Judas and the Black Messiah", "Mank", "The Invisible Man",
        "Onward", "Hamilton", "I'm Thinking of Ending Things", "Emma.", "Greyhound"
    ],
    '2019': [
        "Parasite", "Avengers: Endgame", "Joker", "Once Upon a Time in Hollywood", "1917",
        "Ford v Ferrari", "Jojo Rabbit", "Marriage Story", "The Irishman", "Knives Out",
        "Little Women", "Uncut Gems", "Portrait of a Lady on Fire", "The Lighthouse",
        "Rocketman", "Us", "Midsommar", "Klaus", "Pain and Glory", "Waves"
    ],
    '2018': [
        "Spider-Man: Into the Spider-Verse", "Avengers: Infinity War", "Black Panther", "Roma",
        "Bohemian Rhapsody", "A Star Is Born", "Green Book", "The Favourite", "Hereditary",
        "BlacKkKlansman", "Mission: Impossible - Fallout", "A Quiet Place", "First Man",
        "Isle of Dogs", "Capernaum", "Cold War", "Eighth Grade", "Shoplifters", "Widows", "Annihilation"
    ]
}

class MovieAPIService:
    def __init__(self):
        # Using a public OMDb API key since TMDB is blocked by some ISPs
        self.api_key = "trilogy"
        self.base_url = "http://www.omdbapi.com/"
        self.cache = CacheService(expiration_seconds=86400 * 7)
        self.timeout = 5
        
        # Build synthetic ID mappings for recent movies
        self.recent_id_to_title = {}
        self.recent_title_to_id = {}
        for year_str, titles in RECENT_YEAR_MOVIES.items():
            year_int = int(year_str)
            for idx, title in enumerate(titles):
                synth_id = 900000 + (year_int % 100) * 100 + (idx + 1)
                self.recent_id_to_title[synth_id] = title
                self.recent_title_to_id[title.lower()] = synth_id

        # Load local ML dataset as the primary source for search & model data
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            self.movie_data = joblib.load(os.path.join(base_path, 'models', 'movie_data.pkl'))
        except Exception:
            self.movie_data = None
            
        # Enrich dataset with Vote Count & Popularity from raw data for accurate ranking
        if self.movie_data is not None:
            try:
                raw_path = os.path.join(base_path, 'data', 'raw', 'movies.csv')
                if os.path.exists(raw_path):
                    raw_df = pd.read_csv(raw_path, usecols=['Movie_ID', 'Movie_Vote_Count', 'Movie_Popularity'])
                    self.movie_data = self.movie_data.merge(raw_df, on='Movie_ID', how='left')
            except Exception:
                pass

            self.movie_data['Movie_Vote'] = pd.to_numeric(self.movie_data['Movie_Vote'], errors='coerce').fillna(0)
            if 'Movie_Vote_Count' in self.movie_data.columns:
                self.movie_data['Movie_Vote_Count'] = pd.to_numeric(self.movie_data['Movie_Vote_Count'], errors='coerce').fillna(0)
            else:
                self.movie_data['Movie_Vote_Count'] = 50

            if 'Movie_Popularity' in self.movie_data.columns:
                self.movie_data['Movie_Popularity'] = pd.to_numeric(self.movie_data['Movie_Popularity'], errors='coerce').fillna(0)
            else:
                self.movie_data['Movie_Popularity'] = 10

            # Pre-parse release years
            self.movie_data['Parsed_Year'] = pd.to_datetime(self.movie_data['Movie_Release_Date'], errors='coerce').dt.year

            # Compute Bayesian weighted score for dataset movies
            v = self.movie_data['Movie_Vote_Count']
            r = self.movie_data['Movie_Vote']
            m = 50.0
            c = 6.0
            self.movie_data['rank_score'] = (v / (v + m)) * r + (m / (v + m)) * c + 0.05 * np.log10(v + 1)
            
    def _extract_year(self, date_val, omdb_val=None):
        if omdb_val and str(omdb_val).strip() not in ("N/A", "", "None"):
            m = re.search(r'\b(19\d\d|20\d\d)\b', str(omdb_val))
            if m:
                return m.group(1)
        if date_val and pd.notna(date_val) and str(date_val).strip() not in ("N/A", "", "nan"):
            m = re.search(r'\b(19\d\d|20\d\d)\b', str(date_val))
            if m:
                return m.group(1)
            parts = str(date_val).strip().split('-')
            if len(parts) == 3 and len(parts[-1]) == 4:
                return parts[-1]
            return str(date_val)[:4]
        return "N/A"

    def _fetch_poster_and_details_omdb(self, title):
        cache_key = f"omdb_{title.lower().strip()}"
        cached = self.cache.get(cache_key)
        if cached: 
            return cached
        
        try:
            resp = requests.get(self.base_url, params={"apikey": self.api_key, "t": title}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("Response") == "True":
                self.cache.set(cache_key, data)
                return data
        except Exception:
            pass
        return None

    def _format_movie(self, row=None, omdb_data=None, default_title=None, default_id=None, default_year=None):
        movie_id = default_id or (int(row.get('Movie_ID', 0)) if row is not None else 0)
        title = default_title or (row.get('Movie_Title') if row is not None else 'Unknown')
        raw_date = row.get('Movie_Release_Date', 'N/A') if row is not None else (str(default_year) if default_year else 'N/A')
        release_year = self._extract_year(raw_date)

        movie = {
            "movie_id": movie_id,
            "title": title,
            "poster_url": None,
            "backdrop_url": None,
            "overview": (row.get('Movie_Overview', 'No overview available.') if row is not None else 'Acclaimed feature film.'),
            "release_year": release_year,
            "release_date": raw_date,
            "rating": round(float(row.get('Movie_Vote', 0) if (row is not None and pd.notna(row.get('Movie_Vote'))) else 7.5), 1),
            "vote_count": None,
            "popularity": (row.get('Movie_Popularity', 0) if row is not None else 50),
            "original_language": (row.get('Movie_Language', 'en') if row is not None else 'en'),
            "genre": " • ".join(str(row.get('Movie_Genre', '')).split()[:3]) if row is not None else "Cinema",
            "genres": [g.strip().capitalize() for g in str(row.get('Movie_Genre', '')).split() if g.strip()] if row is not None else [],
            "runtime": (row.get('Movie_Runtime', 120) if row is not None else 120),
            "cast": str(row.get('Movie_Cast', '')) if row is not None else '',
            "director": str(row.get('Movie_Director', '')) if row is not None else '',
            "rated": None,
            "awards": None,
            "box_office": None
        }
        
        # Enhance with OMDb data if available
        if omdb_data:
            if omdb_data.get("Title"):
                movie["title"] = omdb_data.get("Title")

            if omdb_data.get("Year") and omdb_data.get("Year") != "N/A":
                movie["release_year"] = self._extract_year(raw_date, omdb_data.get("Year"))
                
            poster = omdb_data.get("Poster")
            if poster and poster != "N/A":
                movie["poster_url"] = poster
                movie["backdrop_url"] = poster
                
            plot = omdb_data.get("Plot")
            if plot and plot != "N/A":
                movie["overview"] = plot
                
            runtime = omdb_data.get("Runtime")
            if runtime and runtime != "N/A":
                movie["runtime"] = runtime.replace(" min", "").strip()
                
            actors = omdb_data.get("Actors")
            if actors and actors != "N/A":
                movie["cast"] = actors
                
            director = omdb_data.get("Director")
            if director and director != "N/A":
                movie["director"] = director
                
            rating = omdb_data.get("imdbRating")
            if rating and rating != "N/A":
                try:
                    movie["rating"] = float(rating)
                except ValueError:
                    pass

            votes = omdb_data.get("imdbVotes")
            if votes and votes != "N/A":
                movie["vote_count"] = votes

            rated = omdb_data.get("Rated")
            if rated and rated != "N/A":
                movie["rated"] = rated

            awards = omdb_data.get("Awards")
            if awards and awards != "N/A":
                movie["awards"] = awards

            box_office = omdb_data.get("BoxOffice")
            if box_office and box_office != "N/A":
                movie["box_office"] = box_office
                    
            genre = omdb_data.get("Genre")
            if genre and genre != "N/A":
                movie["genre"] = genre
                movie["genres"] = [g.strip() for g in genre.split(",")]

            lang = omdb_data.get("Language")
            if lang and lang != "N/A":
                movie["original_language"] = lang.split(",")[0].strip()
                
        return movie

    def search_movies(self, query):
        if not query:
            return []
            
        results = []
        matched_titles = set()

        # 1. Search recent movies list
        q_lower = query.lower().strip()
        for year_str, titles in RECENT_YEAR_MOVIES.items():
            for t in titles:
                if q_lower in t.lower() and t.lower() not in matched_titles:
                    matched_titles.add(t.lower())
                    synth_id = self.recent_title_to_id.get(t.lower(), 999999)
                    omdb_data = self._fetch_poster_and_details_omdb(t)
                    results.append(self._format_movie(omdb_data=omdb_data, default_title=t, default_id=synth_id, default_year=year_str))
                    if len(results) >= 8:
                        return results

        # 2. Search local DB
        if self.movie_data is not None:
            matches = self.movie_data[self.movie_data['Movie_Title'].str.contains(query, case=False, na=False)].head(8)
            for _, row in matches.iterrows():
                title = row['Movie_Title']
                if title.lower() not in matched_titles:
                    matched_titles.add(title.lower())
                    omdb_data = self._fetch_poster_and_details_omdb(title)
                    results.append(self._format_movie(row, omdb_data))
                    if len(results) >= 8:
                        break
            
        return results

    def get_movie_details(self, movie_id=None, title=None):
        # 1. Check if it's a recent synthetic ID
        if movie_id and int(movie_id) in self.recent_id_to_title:
            real_title = self.recent_id_to_title[int(movie_id)]
            omdb_data = self._fetch_poster_and_details_omdb(real_title)
            return self._format_movie(omdb_data=omdb_data, default_title=real_title, default_id=int(movie_id))

        # 2. Check local dataset
        if self.movie_data is not None:
            if movie_id:
                matches = self.movie_data[self.movie_data['Movie_ID'] == int(movie_id)]
            elif title:
                matches = self.movie_data[self.movie_data['Movie_Title'].str.lower() == str(title).lower().strip()]
            else:
                matches = pd.DataFrame()
                
            if not matches.empty:
                row = matches.iloc[0]
                omdb_data = self._fetch_poster_and_details_omdb(row['Movie_Title'])
                return self._format_movie(row, omdb_data)

        # 3. If title is given and not in dataset, fetch directly from OMDb
        if title:
            omdb_data = self._fetch_poster_and_details_omdb(title)
            if omdb_data:
                synth_id = self.recent_title_to_id.get(title.lower().strip(), 999000)
                return self._format_movie(omdb_data=omdb_data, default_title=title, default_id=synth_id)

        return None

    def filter_movies(self, genre='', year='', language='', limit=20):
        year_str = str(year).strip() if year else ''
        genre_str = str(genre).strip() if genre else ''
        cache_key = f"filter_{genre_str.lower()}_{year_str.lower()}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        results = []

        # Case 1: Year is one of recent curated years (2018 - 2026)
        if year_str in RECENT_YEAR_MOVIES:
            candidate_titles = RECENT_YEAR_MOVIES[year_str]
            
            def fetch_recent(idx_and_title):
                idx, t = idx_and_title
                synth_id = 900000 + (int(year_str) % 100) * 100 + (idx + 1)
                omdb_data = self._fetch_poster_and_details_omdb(t)
                return self._format_movie(omdb_data=omdb_data, default_title=t, default_id=synth_id, default_year=year_str)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                all_fetched = list(executor.map(fetch_recent, enumerate(candidate_titles)))

            # Apply genre filter if provided
            if genre_str:
                filtered = [
                    m for m in all_fetched 
                    if genre_str.lower() in m.get('genre', '').lower() 
                    or any(genre_str.lower() in g.lower() for g in m.get('genres', []))
                ]
                results = filtered[:limit] if filtered else all_fetched[:limit]
            else:
                results = all_fetched[:limit]

            self.cache.set(cache_key, results)
            return results

        # Case 2: Querying from local trained dataset
        if self.movie_data is None:
            return []

        df = self.movie_data.copy()

        # Year filter
        if year_str:
            try:
                if year_str.isdigit():
                    target_year = int(year_str)
                    df = df[df['Parsed_Year'] == target_year]
                elif year_str == '2010s':
                    df = df[(df['Parsed_Year'] >= 2010) & (df['Parsed_Year'] < 2020)]
                elif year_str == '2000s':
                    df = df[(df['Parsed_Year'] >= 2000) & (df['Parsed_Year'] < 2010)]
                elif year_str == '1990s':
                    df = df[(df['Parsed_Year'] >= 1990) & (df['Parsed_Year'] < 2000)]
                elif year_str == '1980s':
                    df = df[(df['Parsed_Year'] >= 1980) & (df['Parsed_Year'] < 1990)]
                elif year_str == 'classic':
                    df = df[(df['Parsed_Year'] < 1980)]
            except Exception:
                pass

        # Genre filter
        if genre_str:
            df = df[df['Movie_Genre'].str.contains(genre_str, case=False, na=False)]

        # Language filter
        if language and 'Movie_Language' in df.columns:
            df = df[df['Movie_Language'].str.lower() == language.lower()]

        # Rank using composite Bayesian score
        if 'rank_score' in df.columns:
            matches = df.sort_values(by='rank_score', ascending=False).head(limit)
        else:
            matches = df.sort_values(by='Movie_Vote', ascending=False).head(limit)

        rows = [row for _, row in matches.iterrows()]

        def fetch_single(row):
            omdb_data = self._fetch_poster_and_details_omdb(row['Movie_Title'])
            return self._format_movie(row, omdb_data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_single, rows))

        self.cache.set(cache_key, results)
        return results

    def get_popular_movies(self, limit=20):
        cache_key = f"popular_movies_top_{limit}"
        cached = self.cache.get(cache_key)
        if cached: 
            return cached
        
        if self.movie_data is None: 
            return []
        
        # Sort by composite Bayesian rank score
        if 'rank_score' in self.movie_data.columns:
            popular = self.movie_data.sort_values(by='rank_score', ascending=False).head(limit)
        else:
            popular = self.movie_data.sort_values(by='Movie_Vote', ascending=False).head(limit)
        
        rows = [row for _, row in popular.iterrows()]
        
        def fetch_single(row):
            omdb_data = self._fetch_poster_and_details_omdb(row['Movie_Title'])
            return self._format_movie(row, omdb_data)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_single, rows))
            
        self.cache.set(cache_key, results)
        return results
        
    def get_trending_movies(self, limit=20):
        return self.get_popular_movies(limit=limit)
