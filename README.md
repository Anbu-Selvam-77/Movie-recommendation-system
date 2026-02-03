🎬 Movie Recommendation Engine

A hybrid content-based movie recommendation system built using TF-IDF and cosine similarity, combining genres, cast, and director metadata, and deployed as an interactive Streamlit web application.

🚀 Project Overview

Traditional recommendation systems often rely only on user ratings.
This project instead uses movie content features to recommend similar movies based on:

🎭 Genres

🎬 Director

👥 Cast (top actors)

The system analyzes textual similarity between movies and suggests the most relevant alternatives.

🧠 Key Features

✅ Hybrid recommendation using Genres + Actors + Director

✅ Feature extraction from raw JSON metadata

✅ TF-IDF Vectorization for text representation

✅ Cosine Similarity for movie similarity computation

✅ Interpretable Similarity Score (%)

✅ Interactive Streamlit UI

✅ Slider to control number of recommendations

✅ Fully self-contained (no external API dependency)

🏗️ System Architecture
Raw Movie Metadata (CSV)
        |
        v
Feature Extraction
(Actors, Director, Genres)
        |
        v
Hybrid Content Construction
        |
        v
TF-IDF Vectorization
        |
        v
Cosine Similarity Matrix
        |
        v
Streamlit Web Application

📂 Project Structure
Movie-Recommendation-Engine/
│
├── app.py                  # Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
├── data/
│   ├── movies.csv
│   └── tmdb_5000_movies.csv
│
└── notebooks/
    └── exploration.ipynb   # Data exploration & experiments

⚙️ Technologies Used

Python

Pandas – data processing

Scikit-learn – TF-IDF & cosine similarity

Streamlit – web application

AST – parsing JSON-like metadata

🧪 How the Recommendation Works

Extract actors, director, and genres from raw metadata

Combine features into a hybrid textual representation

Apply TF-IDF Vectorization

Compute cosine similarity between all movies

Recommend movies with the highest similarity scores

Similarity is displayed as a percentage for transparency.

▶️ How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/your-username/Movie-Recommendation-Engine.git
cd Movie-Recommendation-Engine

2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
python -m streamlit run app.py


Open browser at:

http://localhost:8501

🎯 Example Output

Input Movie: Avatar

Recommendations:

Aliens

Terminator 2: Judgment Day

The Abyss

The Terminator

Guardians of the Galaxy

Each recommendation includes:

Similarity Score (%)

Genres

Director

📈 Future Improvements

🔄 Add collaborative filtering

⭐ Include user ratings

🖼️ Optional poster integration

🌐 Deploy on Streamlit Cloud

📊 Evaluation using Precision@K

🧑‍💻 Author

Anbuselvam N
Data Analytics Enthusiast
