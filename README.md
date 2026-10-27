---
title: CineMatch AI Movie Recommender
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: false
---

# CineMatch: AI-Powered Movie Recommendation & Discovery Engine


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Web Framework](https://img.shields.io/badge/Backend-Flask-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![API](https://img.shields.io/badge/Data-TMDB%20REST%20API-01d277?logo=themoviedatabase&logoColor=white)](https://www.themoviedb.org/)
[![Server](https://img.shields.io/badge/WSGI-Waitress-red)](https://github.com/Pylons/waitress)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end Machine Learning web application that delivers high-accuracy movie recommendations using Content-Based Filtering (TF-IDF & Cosine Similarity) and real-time internet movie discovery powered by the TMDB API.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [License](#-license)

---

## 📖 Overview

Finding the right movie to watch across thousands of options is overwhelming. **CineMatch** solves this challenge by combining a pre-trained **Content-Based Recommendation Model** with **Live Web Search**:

1. **Intelligent Recommendations:** Uses mathematical distance (Cosine Similarity) across 5,000 text features (genres, plot, cast, and director) to find closely matching movies.
2. **Real-Time Discovery:** Connects directly with the TMDB REST API to search internet movies, fetch HD posters, and filter titles by genre (Action, Sci-Fi, Comedy, Thriller, etc.) and release year.

---

## ✨ Key Features

- **🎯 Machine Learning Recommendations:** High-precision suggestions powered by TF-IDF vectorization and pre-calculated cosine similarity matrices.
- **🌐 Real-Time Internet Movie Search:** Live query matching with automatic fallback to TMDB API for newly released films.
- **🔍 Typo-Tolerant Search:** Intelligent fuzzy title matching handles user spelling mistakes gracefully.
- **🎭 Multi-Genre & Year Filtering:** Dedicated filters for Action, Sci-Fi, Drama, Romance, Classics, and modern releases (1980–2026).
- **📊 Transparent Similarity Scores:** Displays match percentage and metadata (release date, vote average, runtime, cast, director).
- **⚡ One-Click Startup:** Includes `run.bat` for automatic `.venv` activation, production server launch, and instant browser opening.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      DATA & ML PIPELINE                     │
│  Dataset (4,800+ Titles) ──► Clean & Preprocess             │
│        ──► Feature Engineering (Genres + Cast + Plot)       │
│        ──► TF-IDF Vectorization (5,000 Dimensions)          │
│        ──► Cosine Similarity Matrix (Precomputed Artifact)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND & DUAL-SOURCE API                 │
│                     Flask + Waitress WSGI                   │
│         ┌────────────────────┴────────────────────┐         │
│         ▼                                         ▼         │
│  ML Similarity Engine                      TMDB REST API    │
│  (Offline .pkl Artifacts)               (Live Search & Cast)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND INTERFACE                     │
│        Responsive Cinematic UI (HTML5, Vanilla CSS, JS)     │
│        Auto-opens in default web browser at localhost:5000   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Machine Learning** | Python, Scikit-Learn, Pandas, NumPy, Joblib |
| **Algorithms** | Content-Based Filtering, TF-IDF Vectorizer, Cosine Similarity |
| **Backend & Server** | Flask, Waitress (Production WSGI) |
| **External APIs** | TMDB (The Movie Database) REST API |
| **Frontend** | HTML5, Vanilla CSS (Glassmorphism / Dark Theme), JavaScript |

---

## 📂 Project Structure

```text
Movie Recommendation System/
├── app/
│   ├── services/
│   │   ├── movie_api_service.py      # TMDB API client & search integration
│   │   └── recommendation_service.py # ML inference & similarity engine
│   ├── static/
│   │   ├── css/style.css             # Cinematic dark theme styling
│   │   └── js/script.js              # Client-side interactions
│   ├── templates/                    # Jinja2 templates (index, movie, about, etc.)
│   └── app.py                        # Flask application routes
├── data/                             # Raw and processed movie datasets
├── models/                           # Serialized ML artifacts (.pkl files)
│   ├── movie_data.pkl                # Preprocessed movie metadata
│   ├── movie_vectorizer.pkl          # Fitted TF-IDF Vectorizer
│   └── similarity_matrix.pkl         # Precomputed Cosine Similarity matrix
├── src/
│   ├── preprocessing.py              # Data cleaning pipeline
│   ├── feature_engineering.py        # Metadata combination & extraction
│   └── train.py                      # Model training & artifact export
├── .env.example                      # Sample environment variables
├── requirements.txt                  # Python dependencies
├── run.bat                           # 1-Click launcher for Windows
└── run.py                            # Waitress WSGI production runner
```

---

## 🚀 Quick Start Guide

### Option 1: One-Click Launch (Windows)

Simply double-click:
```cmd
run.bat
```
*This script automatically activates the `.venv` virtual environment, starts the server, and opens your default browser at `http://127.0.0.1:5000`.*

---

### Option 2: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/cinematch-movie-recommender.git
   cd cinematch-movie-recommender
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   TMDB_API_KEY=your_tmdb_api_key_here
   ```

5. **(Optional) Train or Retrain the Model:**
   ```bash
   python src/train.py
   ```

6. **Start the Application:**
   ```bash
   python run.py
   ```
   Open your browser and visit: **`http://127.0.0.1:5000`**

---

## 🧠 Machine Learning Workflow

1. **Metadata Aggregation:** Combines multiple high-signal textual attributes:
   $$\text{Combined Features} = \text{Genres} + \text{Keywords} + \text{Top Cast} + \text{Director} + \text{Plot Overview}$$
2. **Text Vectorization:** Converts combined text into mathematical vectors using `TfidfVectorizer(max_features=5000, stop_words='english')`.
3. **Similarity Calculation:** Measures the angle between movie vectors using **Cosine Similarity**:
   $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
4. **Fast Inference:** Similarity scores are precomputed into a matrix, enabling sub-15ms recommendation lookups during runtime.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"# CineMatch-AI-Powered-Movie-Recommendation-Discovery-Engine" 
"# CineMatch-AI-Powered-Movie-Recommendation-Discovery-Engine" 
