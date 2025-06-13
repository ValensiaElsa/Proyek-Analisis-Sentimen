# Sentiment Analysis on Tokopedia App Review

Proyek ini adalah model analisis sentimen yang digunakan untuk mengklasifikasikan ulasan menjadi tiga kategori sentimen: positif, negatif, dan netral. Model ini dibangun menggunakan beberapa algoritma klasifikasi, termasuk **Logistic Regression**, **Support Vector Machine (SVM)**, **Voting Classifier**, dan **Deep Learning LSTM**, dengan menggunakan **TF-IDF** (Term Frequency-Inverse Document Frequency) untuk ekstraksi fitur teks.

## Arsitektur Model
Model ini menggunakan beberapa algoritma klasifikasi untuk menganalisis sentimen dari teks ulasan. Algoritma yang digunakan adalah:
- **Logistic Regression**: Model klasifikasi yang sederhana namun efektif untuk teks dengan TF-IDF sebagai representasi fitur.
- **Support Vector Machine (SVM)**: Model klasifikasi yang sangat efektif untuk teks dengan menggunakan kernel yang kuat untuk menangani data yang kompleks.
- **Voting Classifier**: Menggabungkan beberapa model klasifikasi untuk meningkatkan akurasi dengan cara mengambil keputusan berdasarkan mayoritas suara dari beberapa model.
- **Deep Learning LSTM**: Model berbasis **Long Short-Term Memory (LSTM)** untuk menangani teks dalam konteks sekuensial, digunakan pada skema pelatihan 4.

## Dataset
Dataset yang digunakan adalah dataset ulasan yang diambil dari **Google Play Store**. Dataset ini berisi teks ulasan dari aplikasi Tokopedia di Play Store dan digunakan untuk melatih model analisis sentimen. Dataset ini disimpan dalam file `ulasan_tokped.csv` dan digunakan untuk melatih model serta untuk melakukan prediksi sentimen.

### Informasi Dataset:
- **Jumlah Data**: 10.000 ulasan
- **Sumber**: Scraping ulasan dari berbagai produk di **Google Play Store**
- **Kelas Sentimen**: Setiap ulasan dilabeli dengan tiga kategori sentimen:
  - **Positif**
  - **Negatif**
  - **Netral**

### Preprocessing Data:
- **Pembersihan Teks**: Menghilangkan stopwords, tokenisasi, dan lemmatization.
- **Ekstraksi Fitur**: Menggunakan **TF-IDF** untuk mengubah teks menjadi fitur numerik yang dapat digunakan oleh model klasifikasi.

## Skema Pelatihan Model
Proyek ini melibatkan empat skema pelatihan dengan berbagai algoritma klasifikasi dan pembagian data sebagai berikut:

1. **Skema Pelatihan 1** (Logistic Regression, Ekstraksi Fitur: TF-IDF, Pembagian Data: 80/20)
   - Model Logistic Regression dilatih dengan 80% data untuk pelatihan dan 20% data untuk pengujian.

2. **Skema Pelatihan 2** (SVM, Ekstraksi Fitur: TF-IDF, Pembagian Data: 70/30)
   - Model Support Vector Machine dilatih dengan 70% data untuk pelatihan dan 30% data untuk pengujian.

3. **Skema Pelatihan 3** (Voting Classifier, Ekstraksi Fitur: TF-IDF, Pembagian Data: 75/25)
   - Model Voting Classifier dilatih dengan 75% data untuk pelatihan dan 25% data untuk pengujian.

4. **Skema Pelatihan 4** (Deep Learning LSTM, Ekstraksi Fitur: TF-IDF)
   - Model Deep Learning LSTM dilatih menggunakan teknik pembelajaran mendalam dengan representasi fitur TF-IDF.

**Model terbaik yang disimpan adalah model Logistic Regression**, yang memberikan hasil terbaik berdasarkan evaluasi pada data pengujian.

## Format Model yang Disimpan
Model disimpan dalam format berikut:
- **model_logreg.pkl**: Model Logistic Regression yang telah dilatih dan disimpan dalam format pickle untuk digunakan dalam prediksi.
- **tfidf_vectorizer.pkl**: Model TF-IDF Vectorizer yang digunakan untuk mentransformasi data teks.

## Cara Menjalankan

### 1. Menjalankan Proyek di Mesin Lokal
1. **Clone Repository**:
   Clone repository ke mesin lokal Anda dengan perintah:
   ```bash
   git clone https://github.com/ValensiaElsa/Proyek-Analisis-Sentimen.git
   cd Proyek-Analisis-Sentimen

2. **Install Dependencies**:
   Install dependensi yang dibutuhkan dengan menjalankan:

   ```bash
   pip install -r requirements.txt
   ```

3. **Siapkan Dataset**:
   Dataset `ulasan_tokped.csv` sudah ada di dalam repository. Dataset ini berisi **10.000 ulasan** yang diambil dari Google Play Store.

4. **Menjalankan Jupyter Notebook**:
   Untuk menjalankan proyek, buka dan jalankan notebook **Sentiment\_Analysis\_Valensia\_Elsa\_Kurnia.ipynb**:

   ```bash
   jupyter notebook
   ```

   Setelah notebook terbuka di browser, jalankan sel-sel pada notebook untuk melatih dan mengevaluasi model.

5. **Inferensi/Prediksi**:
   Setelah pelatihan selesai, gunakan notebook **Inference\_Valensia\_Elsa\_Kurnia.ipynb** untuk melakukan prediksi pada ulasan baru.

### 2. Menjalankan Proyek di Google Colab

1. **Upload Repository ke Google Drive**:
   Upload folder proyek ke Google Drive Anda.

2. **Mount Google Drive di Colab**:
   Buka Google Colab dan mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Install Dependencies**:
   Install dependensi yang dibutuhkan di Colab dengan perintah:

   ```python
   !pip install -r /content/drive/MyDrive/Proyek-Analisis-Sentimen/requirements.txt
   ```

4. **Load Dataset**:
   Pastikan dataset `ulasan_tokped.csv` berada di path yang benar. Anda bisa memuatnya menggunakan `pandas`:

   ```python
   import pandas as pd
   data = pd.read_csv('/content/drive/MyDrive/Proyek-Analisis-Sentimen/ulasan_tokped.csv')
   print(data.head())
   ```

5. **Melatih Model**:
   Jalankan notebook **Sentiment\_Analysis\_Valensia\_Elsa\_Kurnia.ipynb** di Google Colab untuk melatih model dengan dataset.

6. **Prediksi**:
   Setelah model dilatih, jalankan notebook **Inference\_Valensia\_Elsa\_Kurnia.ipynb** untuk melakukan prediksi pada ulasan baru.


