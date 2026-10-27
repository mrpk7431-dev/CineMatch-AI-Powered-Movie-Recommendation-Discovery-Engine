# CineMatch: AI-Powered Movie Recommendation & Discovery Engine

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online%20on%20Render-00c7b7?style=for-the-badge&logo=render&logoColor=white)](https://cinematch-ai-smt8.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Backend](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![API](https://img.shields.io/badge/Data-TMDB%20API-01d277?style=for-the-badge&logo=themoviedatabase&logoColor=white)](https://www.themoviedb.org/)

**[🌐 Click Here to Open Live Application](https://cinematch-ai-smt8.onrender.com/)**

</div>

> **CineMatch** is an end-to-end Machine Learning web application that delivers high-precision, personalized movie recommendations using Content-Based Filtering (TF-IDF & Cosine Similarity), combined with real-time internet movie discovery powered by the TMDB REST API.

---

## 📌 Table of Contents
- [Live Demo](#-live-demo)
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Deployment](#-deployment)
- [License](#-license)

---

## 🌐 Live Demo

The application is deployed and live in production on Render:

| Service | Link | Status |
|---|---|---|
| **Production Web App** | [https://cinematch-ai-smt8.onrender.com/](https://cinematch-ai-smt8.onrender.com/) | ![Operational](https://img.shields.io/badge/Status-Operational-brightgreen) |

> *Note: On Render's free tier, the service may take ~30 seconds to wake up if inactive.*

---

## 📖 Overview

Finding the right movie across thousands of titles can be overwhelming. **CineMatch** solves this challenge by combining a pre-trained **Content-Based Recommendation Engine** with **Live Web Discovery**:

1. **Intelligent Recommendations:** Uses mathematical distance (**Cosine Similarity**) across **5,000 text features** (genres, plot synopsis, keywords, cast, and director) to find closely matching movies in under 15ms.
2. **Real-Time Discovery:** Connects directly with the **TMDB REST API** to search internet movies, fetch HD posters, and filter titles by genre (Action, Sci-Fi, Comedy, Thriller, etc.) and release year (1980–2026).

---

## ✨ Key Features

- **🎯 Machine Learning Recommendations:** High-precision suggestions powered by TF-IDF vectorization and pre-calculated cosine similarity matrices.
- **🌐 Real-Time Internet Movie Search:** Live query matching with automatic fallback to TMDB API for newly released films.
- **🔍 Typo-Tolerant Search:** Intelligent fuzzy title matching handles user spelling mistakes gracefully.
- **🎭 Multi-Genre & Year Filtering:** Dedicated filters for Action, Sci-Fi, Drama, Romance, Classics, and modern releases (1980–2026).
- **📊 Transparent Similarity Scores:** Displays match percentage and metadata (release date, vote average, runtime, cast, director).
- **⚡ One-Click Local Startup:** Includes `run.bat` for automatic `.venv` activation, production server launch, and instant browser opening.

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
| **Frontend** | HTML5, Vanilla CSS (Cinematic Dark Theme / Glassmorphism), JavaScript |
| **Deployment** | Render Cloud, Git, GitHub |

---

## 🧠 Machine Learning Pipeline

1. **Metadata Aggregation:** Combines multiple high-signal textual attributes into a unified representation:
   $$\text{Combined Features} = \text{Genres} + \text{Keywords} + \text{Top Cast} + \text{Director} + \text{Plot Overview}$$
2. **Text Vectorization:** Converts combined text into mathematical vectors using `TfidfVectorizer(max_features=5000, stop_words='english')`.
3. **Similarity Calculation:** Measures the angle between movie vectors using **Cosine Similarity**:
   $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
4. **Fast Inference:** Similarity scores are precomputed into a matrix, enabling sub-15ms recommendation lookups during runtime.

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
   git clone https://github.com/mrpk7431-dev/CineMatch-AI-Powered-Movie-Recommendation-Discovery-Engine.git
   cd CineMatch-AI-Powered-Movie-Recommendation-Discovery-Engine
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

5. **Start the Application:**
   ```bash
   python run.py
   ```
   Open your browser and visit: **`http://127.0.0.1:5000`**

---

## ☁️ Deployment

This project is configured for cloud hosting on **Render**:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python run.py`
- **Environment Variable:** `TMDB_API_KEY`

`run.py` dynamically binds to `0.0.0.0` and the port assigned by the hosting environment (`$PORT`), allowing seamless deployment across any cloud provider.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
