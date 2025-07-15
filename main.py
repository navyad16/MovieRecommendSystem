import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")  # should have columns like title, genre, director, cast, overview

# Strip whitespace and convert to lowercase (optional)
movies.columns = movies.columns.str.strip()

# Fill missing values with empty string
for feature in ['genre', 'director', 'cast', 'overview']:
    movies[feature] = movies[feature].fillna('')

# Combine features into a single string
def combine_features(row):
    return row['genre'] + " " + row['director'] + " " + row['cast'] + " " + row['overview']

movies["combined_features"] = movies.apply(combine_features, axis=1)

# Convert text to vector
vectorizer = CountVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(movies["combined_features"])

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

# Function to recommend movies
def recommend_movie(movie_title):
    movie_title = movie_title.lower()
    if movie_title not in movies['title'].str.lower().values:
        print("Movie not found in database.")
        return

    index = movies[movies['title'].str.lower() == movie_title].index[0]
    similar_movies = list(enumerate(similarity[index]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\nTop 5 movies similar to '{movie_title.title()}':")
    for i in sorted_movies:
        print(movies.iloc[i[0]].title)

# Example usage
movie_input = input("Enter a movie name: ")
recommend_movie(movie_input)