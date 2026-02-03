import streamlit as st
import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation Engine",
    layout="wide"
)

st.title("🎬 Movie Recommendation Engine")
st.write("Hybrid recommender using **Genres + Actors + Director**")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    tmdb = pd.read_csv("data/tmdb_5000_movies.csv")
    return movies, tmdb

movies, tmdb = load_data()

# -------------------------------------------------
# FEATURE EXTRACTION FUNCTIONS
# -------------------------------------------------
def extract_actors(cast):
    try:
        cast_list = ast.literal_eval(cast)
        return " ".join([c["name"].lower() for c in cast_list[:5]])
    except:
        return ""

def extract_director(crew):
    try:
        crew_list = ast.literal_eval(crew)
        for member in crew_list:
            if member.get("job") == "Director":
                return member.get("name", "").lower()
        return ""
    except:
        return ""

def extract_genres(genres):
    try:
        genre_list = ast.literal_eval(genres)
        return " ".join([g["name"].lower() for g in genre_list])
    except:
        return ""

# -------------------------------------------------
# BUILD FEATURES
# -------------------------------------------------
movies["actors"] = movies["cast"].apply(extract_actors)
movies["director"] = movies["crew"].apply(extract_director)

tmdb["genres_clean"] = tmdb["genres"].apply(extract_genres)

movies = movies.merge(
    tmdb[["title", "genres_clean"]],
    on="title",
    how="left"
)

movies["genres_clean"] = movies["genres_clean"].fillna("")

# -------------------------------------------------
# HYBRID CONTENT (WEIGHTED)
# -------------------------------------------------
movies["hybrid_content"] = (
    movies["actors"] + " " +
    movies["actors"] + " " +          # actors ×2
    movies["director"] + " " +
    movies["director"] + " " +
    movies["director"] + " " +        # director ×3
    movies["genres_clean"] + " " +
    movies["genres_clean"]            # genres ×2
)

# -------------------------------------------------
# BUILD SIMILARITY MODEL
# -------------------------------------------------
@st.cache_resource
def build_model(text):
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
    tfidf_matrix = tfidf.fit_transform(text)
    return cosine_similarity(tfidf_matrix)

cosine_sim = build_model(movies["hybrid_content"])

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# -------------------------------------------------
# RECOMMENDATION FUNCTION (WITH SIMILARITY %)
# -------------------------------------------------
def recommend_movies(title, top_n=6):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [round(i[1] * 100, 2) for i in sim_scores]

    results = movies.loc[movie_indices, ["title", "genres_clean", "director"]]
    results["similarity"] = similarity_scores

    return results

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
movie_selected = st.selectbox(
    "🎥 Select a movie",
    sorted(movies["title"].values)
)

top_n = st.slider(
    "🔢 Number of recommendations",
    min_value=3,
    max_value=12,
    value=6
)

if st.button("🎯 Recommend"):
    st.subheader("Recommended Movies")

    results = recommend_movies(movie_selected, top_n)
    cols = st.columns(3)

    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 3]:
            st.image(
                "https://via.placeholder.com/300x450?text=Movie",
                use_container_width=True
            )
            st.markdown(f"### 🎬 {row['title']}")
            st.markdown(f"**🔢 Similarity:** {row['similarity']}%")
            st.markdown(f"**🎭 Genres:** {row['genres_clean']}")
            st.markdown(f"**🎬 Director:** {row['director']}")
