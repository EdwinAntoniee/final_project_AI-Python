import streamlit as st
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

TMDB_API_KEY = "09064ce99d3e0ce44969ac60940cfe5a"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_movie_poster(movie_title, year=None):
    try:
        search_url = f"{TMDB_BASE_URL}/search/movie"
        response = requests.get(search_url, params={
            "api_key": TMDB_API_KEY,
            "query": movie_title,
            "year": year
        })
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0]["poster_path"]
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    except Exception as e:
        print(f"Error getting poster for {movie_title}: {str(e)}")
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"
st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_movie_data():
    try:
        df = pd.read_csv("movies.csv", quoting=1)
        required_columns = ['title', 'year', 'genre', 'description', 'rating']
        if not all(col in df.columns for col in required_columns):
            st.error("Dataset tidak memiliki kolom yang diperlukan!")
            return pd.DataFrame(columns=required_columns)
        df['genre'] = df['genre'].fillna('')
        df['description'] = df['description'].fillna('')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.drop_duplicates(subset=['title'], keep='first')
        df = df.dropna(subset=['title', 'genre'])
        if df.empty:
            st.error("Dataset kosong setelah pembersihan data!")
            return pd.DataFrame(columns=required_columns)
        return df
    except pd.errors.ParserError as e:
        st.error(f"Error membaca file CSV: Format tidak sesuai. Pastikan setiap kolom dipisahkan dengan benar.\nDetail: {str(e)}")
        return pd.DataFrame(columns=required_columns)
    except FileNotFoundError:
        st.error("File movies.csv tidak ditemukan! Pastikan file berada di folder yang sama dengan app.py")
        return pd.DataFrame(columns=required_columns)
    except Exception as e:
        st.error(f"Error tidak terduga saat memuat dataset: {str(e)}")
        return pd.DataFrame(columns=required_columns)

def get_mood_from_openrouter(text):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = "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"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "OpenRouter-Marketplace": "true"
    }
    text_lower = text.lower()
    keyword_mapping = {
        'bosan': ['bosan', 'jenuh', 'monoton', 'capek', 'rutinitas'],
        'sedih': ['sedih', 'galau', 'kecewa', 'murung', 'patah hati'],
        'senang': ['senang', 'bahagia', 'gembira', 'suka', 'ceria'],
        'semangat': ['semangat', 'antusias', 'energik', 'excited'],
        'takut': ['takut', 'cemas', 'khawatir', 'ngeri'],
        'penasaran': ['penasaran', 'ingin tahu', 'curious'],
        'marah': ['marah', 'kesal', 'jengkel', 'emosi'],
        'cinta': ['cinta', 'sayang', 'romantis', 'love'],
        'tegang': ['tegang', 'stress', 'tertekan', 'pressure']
    }
    for mood, keywords in keyword_mapping.items():
        if any(keyword in text_lower for keyword in keywords):
            return mood
    prompt = f"""Analisis mood dari teks berikut ini. Pilih satu mood yang paling tepat:
    bosan = jika terkait kebosanan, kejenuhan, rutinitas
    sedih = jika terkait kesedihan, kekecewaan
    senang = jika terkait kebahagiaan, keceriaan
    semangat = jika terkait antusiasme, energi
    takut = jika terkait ketakutan, kecemasan
    penasaran = jika terkait rasa ingin tahu
    marah = jika terkait kemarahan, kejengkelan
    cinta = jika terkait perasaan romantis
    tegang = jika terkait ketegangan, stress
    Berikan jawaban dalam satu kata saja.
    Teks: {text}
    Mood:"""
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "system", "content": "Kamu adalah ahli analisis emosi yang selalu memberikan jawaban singkat satu kata."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10,
                "temperature": 0.1,
                "stop": ["\n", ".", ",", "!", "?"]
            }
        )
        if response.status_code == 200:
            result = response.json()
            mood = result['choices'][0]['message']['content'].strip().lower()
            if mood in keyword_mapping:
                return mood
            for valid_mood in keyword_mapping.keys():
                if valid_mood in mood:
                    return valid_mood
    except Exception as e:
        st.warning(f"Error saat menganalisis mood: {str(e)}")
    if 'capek' in text_lower or 'rutinitas' in text_lower:
        return 'bosan'
    return 'bosan'  
def classify_text_to_genre(text):
    mood_mapping = {
        'senang': ['Comedy', 'Adventure', 'Animation'],
        'sedih': ['Drama', 'Romance'],
        'semangat': ['Action', 'Adventure', 'Sport'],
        'takut': ['Horror', 'Thriller'],
        'penasaran': ['Mystery', 'Crime', 'Thriller'],
        'marah': ['Action', 'Crime', 'Drama'],
        'bosan': ['Adventure', 'Fantasy', 'Sci-Fi'],
        'cinta': ['Romance', 'Drama', 'Comedy'],
        'tegang': ['Thriller', 'Mystery', 'Crime']
    }
    mood = get_mood_from_openrouter(text)
    genres = mood_mapping.get(mood, ['Drama', 'Action'])
    st.write(f"Mood terdeteksi: {mood}")
    st.write(f"Genre yang direkomendasikan: {', '.join(genres)}")
    return genres

def get_recommendations_from_text(user_text, df):
    genres = classify_text_to_genre(user_text)
    if genres:
        masks = [df['genre'].str.contains(genre, case=False) for genre in genres]
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask | mask
        recommendations = df[final_mask].copy()
        recommendations['genre_match'] = 0
        for genre in genres:
            recommendations.loc[recommendations['genre'].str.contains(genre, case=False), 'genre_match'] += 1
        return recommendations.sort_values(['genre_match', 'rating'], ascending=[False, False]).head(5)
    return pd.DataFrame()

def get_similar_movies(movie_title, df):
    if movie_title not in df['title'].values:
        return pd.DataFrame()
    tfidf = TfidfVectorizer(stop_words='english')
    df['combined_features'] = (df['genre'].str.lower() + ' ' + 
                             df['genre'].str.lower() + ' ' + 
                             df['genre'].str.lower() + ' ' + 
                             df['description'].str.lower())
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    filtered_scores = [i for i in sim_scores[1:] if i[1] > 0.3]
    if not filtered_scores:
        return pd.DataFrame()
    movie_indices = [i[0] for i in filtered_scores[:3]]  
    similar_movies = df.iloc[movie_indices].copy()
    similar_movies['similarity_score'] = [i[1] for i in filtered_scores[:3]] 
    similar_movies['combined_score'] = (
        0.7 * similar_movies['similarity_score'] + 
        0.3 * (similar_movies['rating'] / 10.0)
    )
    return similar_movies.sort_values('combined_score', ascending=False)

def get_recommendations_from_preferences(mood, purpose, genres, categories, year_range, df):
    year_ranges = {
        "Film Terbaru (2020+)": (2020, 2025),
        "Film 5-10 Tahun Terakhir (2015-2020)": (2015, 2020),
        "Film Klasik (2000-2015)": (2000, 2015),
        "Film Lawas (Sebelum 2000)": (1900, 2000)
    }
    min_year, max_year = year_ranges[year_range]
    filtered_df = df[(df['year'] >= min_year) & (df['year'] <= max_year)].copy()
    filtered_df['score'] = 0
    for genre in genres:
        filtered_df.loc[filtered_df['genre'].str.contains(genre, case=False), 'score'] += 2
    for category in categories:
        filtered_df.loc[filtered_df['description'].str.contains(category, case=False), 'score'] += 1
    mood_genres = {
        "Senang": ["Comedy", "Romance", "Adventure"],
        "Sedih": ["Drama", "Romance"],
        "Bosan": ["Action", "Adventure", "Sci-Fi"],
        "Semangat": ["Action", "Sport", "Adventure"],
        "Penasaran": ["Mystery", "Thriller", "Crime"]
    }
    purpose_genres = {
        "Nonton sendirian": ["Drama", "Thriller", "Mystery"],
        "Keluarga": ["Animation", "Adventure", "Family"],
        "Pasangan": ["Romance", "Comedy", "Drama"],
        "Teman": ["Action", "Comedy", "Horror"]
    }
    if mood in mood_genres:
        for genre in mood_genres[mood]:
            filtered_df.loc[filtered_df['genre'].str.contains(genre, case=False), 'score'] += 1.5
    if purpose in purpose_genres:
        for genre in purpose_genres[purpose]:
            filtered_df.loc[filtered_df['genre'].str.contains(genre, case=False), 'score'] += 1.5
    return filtered_df.sort_values(['score', 'rating'], ascending=[False, False]).head(5)

movies_df = load_movie_data()

def display_movie_recommendations(recommendations):
    if not recommendations.empty:
        cols = st.columns(3)
        for idx, (_, movie) in enumerate(recommendations.iterrows()):
            with cols[idx % 3]:
                poster_url = get_movie_poster(movie['title'], str(movie['year']))
                st.image(poster_url, use_container_width=True)
                st.markdown(f"<h3 style='text-align: center;'>{movie['title']} ({movie['year']})</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Rating: {movie['rating']}/10</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Genre: {movie['genre']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>{movie['description'][:150]}...</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center;'>Tidak ditemukan film yang sesuai. Coba kata kunci lain!</p>", unsafe_allow_html=True)

def main():
    col1, col2, col3 = st.columns([2,1,2])    
    with col2:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXdpZnhycWQ5dHVxcmo3aTVwb3Byaml0a3V4eWE1dDB6OTZ0b3FmZyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/26DOMQa5Ib2SmaRZm/giphy.gif", 
                 use_container_width=True,
                 caption="Movie Recommendation System")
    st.markdown("<h1 style='text-align: center;'>🎬 Sistem Rekomendasi Film</h1>", unsafe_allow_html=True)
    all_genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", 
                 "Drama", "Family", "Fantasy", "Horror", "Mystery",
                 "Romance", "Sci-Fi", "Thriller"]
    col1, col2, col3 = st.columns([1, 2, 1])
    df = load_movie_data()
    if df.empty:
        st.markdown("<p style='text-align: center;'>Tidak dapat memuat data film. Silakan periksa file movies.csv</p>", unsafe_allow_html=True)
        return
    tab1, tab2, tab3, tab4 = st.tabs([
        "💭 Rekomendasi dari Mood", 
        "🎯 Film Serupa",
        "📝 Kuesioner Lengkap",
        "ℹ️ Tentang"
    ])
    with tab1:
        st.markdown("<h2 style='text-align: center;'>Rekomendasi Film Berdasarkan Mood</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Ceritakan perasaan atau suasana hati Anda, dan saya akan merekomendasikan film yang cocok!</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_input = st.text_area("Bagaimana perasaan Anda hari ini?", height=100)
            if st.button("Dapatkan Rekomendasi", key="mood_button", use_container_width=True):
                if user_input:
                    with st.spinner("Mencari film yang cocok untuk Anda..."):
                        st.markdown("<h3 style='text-align: center;'>Film film ini cocok untuk suasana hati Anda:</h3>", unsafe_allow_html=True)
                        recommendations = get_recommendations_from_text(user_input, df)
                        display_movie_recommendations(recommendations)
                else:
                    st.markdown("<p style='text-align: center;'>Mohon ceritakan perasaan Anda terlebih dahulu!</p>", unsafe_allow_html=True)
    with tab2:
        st.markdown("<h2 style='text-align: center;'>Temukan Film Serupa</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Pilih film yang Anda sukai, dan saya akan merekomendasikan film serupa!</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            selected_movie = st.selectbox(
                "Pilih film yang Anda sukai:",
                df['title'].tolist()
            )
            if st.button("Cari Film Serupa", key="similar_button", use_container_width=True):
                with st.spinner("Mencari film serupa..."):
                    similar_movies = get_similar_movies(selected_movie, df)
                    display_movie_recommendations(similar_movies)
    with tab3:
        st.markdown("<h2 style='text-align: center;'>Kuesioner Rekomendasi Film</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Mari kita cari film yang cocok untuk Anda dengan menjawab beberapa pertanyaan berikut:</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            mood = st.selectbox(
                "Bagaimana suasana hati Anda saat ini?",
                ["Senang", "Sedih", "Bosan", "Semangat", "Penasaran"],
                key="mood_select"
            )
            purpose = st.selectbox(
                "Dengan siapa Anda berencana menonton?",
                ["Nonton sendirian", "Keluarga", "Pasangan", "Teman"],
                key="purpose_select"
            )
            genres = st.multiselect(
                "Pilih genre film yang Anda minati (maksimal 3):",
                all_genres,
                max_selections=3,
                key="genre_select"
            )
            categories = st.multiselect(
                "Tema atau elemen apa yang Anda sukai dalam film?",
                ["Superhero", "Time Travel", "Musical", "Based on True Story",
                 "Friendship", "Space", "Magic", "Sports", "War", "History"],
                max_selections=3,
                key="category_select"
            )
            year_range = st.selectbox(
                "Rentang tahun film yang diinginkan:",
                ["Film Terbaru (2020+)",
                 "Film 5-10 Tahun Terakhir (2015-2020)",
                 "Film Klasik (2000-2015)",
                 "Film Lawas (Sebelum 2000)"],
                key="year_select"
            )
            if st.button("Dapatkan Rekomendasi", key="questionnaire_button", use_container_width=True):
                if not genres:
                    st.markdown("<p style='text-align: center;'>Mohon pilih setidaknya satu genre film!</p>", unsafe_allow_html=True)
                else:
                    with st.spinner("Mencari rekomendasi film yang sesuai..."):
                        recommendations = get_recommendations_from_preferences(
                            mood, purpose, genres, categories, year_range, df
                        )
                        display_movie_recommendations(recommendations)
    with tab4:
        st.header("Tentang Aplikasi")
        st.write("""
        Aplikasi sistem rekomendasi film ini dirancang untuk memberikan saran film yang personal dan relevan melalui beberapa pendekatan utama:

        1. **Rekomendasi Berdasarkan Mood**: Menganalisis teks yang Anda tulis untuk memahami suasana hati, lalu merekomendasikan film yang sesuai.
        2. **Film Serupa**: Menemukan film dengan karakteristik mirip berdasarkan genre dan deskripsi film yang Anda sukai.
        3. **Kuesioner Lengkap**: Memberikan rekomendasi berdasarkan preferensi detail Anda melalui serangkaian pertanyaan.

        Koleksi film yang digunakan mencakup judul-judul internasional dengan deskripsi berbahasa Inggris untuk memastikan analisis yang akurat. Seluruh antarmuka aplikasi disajikan dalam bahasa Indonesia agar mudah digunakan oleh semua kalangan.
        """)

if __name__ == "__main__":
    main()