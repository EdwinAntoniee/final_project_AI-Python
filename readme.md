# 🎬 CineMatch - Movie Recommender App

CineMatch adalah aplikasi rekomendasi film berbasis AI yang membantu pengguna menemukan film sesuai suasana hati, tujuan menonton, dan preferensi genre. Aplikasi ini menggunakan data dari TMDB API dan model AI untuk mendeteksi mood dari input pengguna.

---

## 📌 Fitur Utama

### 1. Mood Detection via Chatbot
- Ketikkan perasaan seperti "Aku ingin tertawa bahagia".
- Sistem akan mendeteksi mood dengan OpenRouter AI dan menampilkan film relevan.

### 2. Similar Movie Recommendation
- Masukkan judul film favoritmu.
- Aplikasi akan merekomendasikan film dengan genre dan nuansa serupa.

### 3. Kuesioner Lengkap
- Jawab beberapa pertanyaan interaktif seputar mood, genre favorit, tujuan menonton, dan lainnya.
- Sistem akan menampilkan film yang sesuai secara personal.

---

## ⚙️ Cara Menjalankan di Lokal

### 1. Persyaratan Sistem
- OS: Windows 10/11 (direkomendasikan), Linux/macOS (dengan penyesuaian path)
- Python: Versi 3.8 ke atas
- API:
  - TMDB API Key (untuk data film & poster)
  - OpenRouter API Key (untuk analisis mood)

### 2. Library yang Dibutuhkan
- `streamlit`
- `pandas`
- `requests`
- `scikit-learn`
- `transformers`
- `warnings` (built-in Python)

> Install semua dependensi melalui file `requirements.txt`.

### 3. Langkah Instalasi

```bash
# Clone repository
git clone https://github.com/EdwinAntoniee/final_project_AI-Python.git
cd final_project_AI-Python

# (Opsional) Buat virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

