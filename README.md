# 🎥 Movie Recommendation System

A highly optimized, content-based Movie Recommendation System built with Python, scikit-learn, and Pandas. This project analyzes movie features like **Genres**, **Keywords**, **Cast Members**, and **Directors** using Nearest Neighbors to suggest movies similar to your favorites.

## ✨ Features

- **TF-IDF Vectorization:** Analyzes textual movie data using Term Frequency-Inverse Document Frequency to highlight unique characteristics over generic tagging.
- **Top Cast Extraction:** Intelligently extracts the top 3 actors of each movie to heavily weigh stars in finding similar content.
- **Fuzzy Search:** Includes typo-tolerance. Searching `batmn` will recognize you likely meant `Batman` and offer suggestions.
- **Lightning Fast:** Uses pre-calculated `.pkl` (Pickle) models and sparse matrices to load data instantly and use minimal RAM.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed, then install the core dependencies:

```bash
pip install pandas scikit-learn
```

### 1. Training the Model

Before running the application, you need to generate the machine learning models. Ensure that `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` are placed in the root directory, then run:

```bash
python recommender.py
```

This script will extract the data, train the KNN model, and generate a `/model` directory containing `knn_model.pkl`, `vectorizer.pkl`, and `movies.pkl`.

### 2. Running the Application

Once the models are generated, you can launch the interactive terminal application:

```bash
python project.py
```

Simply type the name of a movie you enjoy (e.g., `Avatar`), and the system will output the top 5 highly-accurate recommendations along with their similarity scores!