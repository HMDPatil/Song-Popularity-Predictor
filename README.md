# Spotify Song Popularity Predictor 🎵

This is my first machine learning project!  
It predicts the **popularity of a song (0–100)** using basic **linear regression** from scratch in NumPy.

## 📂 Dataset
- Dataset used: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-dataset-19212020-160k-tracks)
- Cleaned to use only 7 numerical features:
  - Danceability
  - Energy
  - Loudness
  - Tempo
  - Duration (ms)
  - Valence
  - Popularity (target)

## 📊 Features
- Implemented Linear Regression using NumPy
- Normalized input features
- Used gradient descent (no ML libraries like Scikit-learn)
- Trained on full dataset (~232k songs)
- Plotted cost vs iterations
- Calculated R² score to measure performance
- Predict song popularity with user input

## ⚠️ Limitations
- Linear regression is too simple for this task
- R² score ~ 0.17 (weak performance, but expected)
- Useful as a **learning project**

## 🛠 How to Run
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`
- Run the file:
```bash
python song_popularity.py
