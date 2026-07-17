# app.py -- lancer avec : streamlit run app.py
import re
import streamlit as st
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 200
INDEX_FROM = 3
MODEL_PATH = 'imdb_sentiment.keras'


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_word_index():
    return imdb.get_word_index()


def preprocess_text(text, word_index):
    # meme encodage qu'a l'entrainement : token de debut (1), decalage +3, OOV=2, padding 'pre'
    tokens = [1]
    for w in re.findall(r"[a-z']+", text.lower()):
        idx = word_index.get(w)
        if idx is None or idx + INDEX_FROM >= VOCAB_SIZE:
            tokens.append(2)
        else:
            tokens.append(idx + INDEX_FROM)
    return pad_sequences([tokens], maxlen=MAX_LEN, padding='pre', truncating='pre')


model = load_model()
word_index = load_word_index()

st.title('Analyse de sentiment IMDB')
st.write('Saisissez une critique de film (en anglais) : le modele predit si elle est positive ou negative.')
st.caption('Note : seuls les 200 premiers mots sont analyses (le reste est tronque).')

user_input = st.text_area('Votre critique', height=150)

if st.button('Analyser'):
    if not user_input.strip():
        st.warning('Veuillez saisir une critique.')
    else:
        proba = float(model.predict(preprocess_text(user_input, word_index), verbose=0)[0, 0])
        if proba > 0.5:
            st.success(f'Positif  (confiance {proba:.1%})')
        else:
            st.error(f'Negatif  (confiance {1 - proba:.1%})')
        st.write(f'Score brut : {proba:.4f}')
