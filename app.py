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

EXAMPLES = {
    'Exemple positif': (
        "This film is an absolute masterpiece. The acting was brilliant, the story deeply "
        "moving, and every scene beautifully shot. I was captivated from start to finish. "
        "One of the best movies I have ever seen."
    ),
    'Exemple negatif': (
        "What a complete waste of time. The plot made no sense, the acting was wooden and "
        "unconvincing, and the pacing was painfully slow. I was bored the entire time. "
        "Easily one of the worst films of the year."
    ),
    'Exemple mitige': (
        "The movie had some good moments and the visuals were stunning, but the story dragged "
        "in the middle and the ending felt rushed. A decent watch but nothing memorable."
    ),
}


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

st.set_page_config(page_title='Sentiment IMDB')
st.title('Analyse de sentiment IMDB')
st.write(
    'Collez une critique de film **en anglais** : un LSTM bidirectionnel (~87 % de précision) '
    'prédit si elle est positive ou négative.'
)

if 'review' not in st.session_state:
    st.session_state.review = ''

st.write('**Essayez un exemple :**')
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.review = text

st.text_area('Votre critique', height=180, key='review')
st.caption(
    'Le modèle est entraîné sur des critiques longues : il est bien plus sûr sur un vrai '
    'paragraphe que sur « Great! ». Seuls les 200 premiers mots sont analysés.'
)

if st.button('Analyser', type='primary'):
    review = st.session_state.review
    if not review.strip():
        st.warning('Veuillez saisir une critique.')
    else:
        proba = float(model.predict(preprocess_text(review, word_index), verbose=0)[0, 0])
        confidence = proba if proba > 0.5 else 1 - proba
        if proba > 0.5:
            st.success('Positif')
        else:
            st.error('Négatif')
        st.metric('Confiance', f'{confidence:.1%}')
        st.progress(confidence)
        st.caption(f'Score brut : {proba:.4f}  (0 = négatif, 1 = positif)')
