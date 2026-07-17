# Deep Learning J4 — RNN, LSTM, GRU

Réseaux de neurones récurrents (RNN / LSTM / GRU) appliqués aux **séries temporelles** et au **NLP**, avec déploiement d'une **WebApp Streamlit**. Le tout tourne en local via Docker (TensorFlow 2.18 / Keras 3).

## Les projets

### 1. Séries temporelles — Airline Passengers · `airline_passengers.ipynb`
Prévision du trafic aérien mensuel (144 points, 1949-1960) via une fenêtre glissante (12 mois → le 13ᵉ).
- **LSTM** : RMSE test **35 passagers** (objectif < 50).
- Comparaison **LSTM vs GRU** (détail dans `results.md`) : le GRU obtient un RMSE équivalent avec **~24 % de paramètres en moins**.

### 2. NLP — classification de texte
- **IMDB** · `imdb_sentiment.ipynb` : sentiment positif/négatif de critiques de films, **LSTM bidirectionnel** → **accuracy 86,8 %** (F1 0,86). Modèle sauvegardé : `imdb_sentiment.keras`.
- **AG News** · `ag_news.ipynb` : titres de presse classés en 4 catégories (World / Sports / Business / Sci/Tech), sortie **softmax multiclasse** → **accuracy 91,3 %**.

### 3. WebApp — `app.py`
Interface Streamlit qui charge le modèle IMDB et prédit le sentiment d'une critique saisie en direct.

## Lancer

Prérequis : **Docker**.

**Notebooks (Jupyter + TensorBoard) :**
```bash
docker compose up
```
- Jupyter : http://localhost:8888
- TensorBoard : http://localhost:6006

**WebApp Streamlit :**
```bash
docker compose --profile webapp up
```
- App : http://localhost:8501

## Résultats

| Modèle | Tâche | Métrique |
|--------|-------|----------|
| LSTM | Airline Passengers (régression) | RMSE test **35** passagers |
| GRU | Airline Passengers (régression) | RMSE test **33** passagers · −24 % de params |
| Bi-LSTM | IMDB (binaire) | accuracy **86,8 %** · F1 0,86 |
| LSTM | AG News (4 classes) | accuracy **91,3 %** |

## Structure

| Fichier | Rôle |
|---------|------|
| `airline_passengers.ipynb` | Séries temporelles (LSTM + comparaison GRU) |
| `imdb_sentiment.ipynb` | Sentiment IMDB (Bi-LSTM) |
| `ag_news.ipynb` | Classification multiclasse AG News |
| `app.py` | WebApp Streamlit (inférence IMDB) |
| `results.md` | Tableau comparatif LSTM vs GRU |
| `*.keras` | Modèles entraînés |
| `Dockerfile` · `docker-compose.yml` · `requirements.txt` | Environnement |

## Stack

Python 3.11 · TensorFlow 2.18 · Keras 3 · scikit-learn · Streamlit · HuggingFace datasets
