import streamlit as st, os, time, csv, pickle, pinecone
from streamlit_extras.stateful_button import button

from pinecone import Pinecone

from utils import st_def, tmdb, mypinecone, docs

st_def.st_logo(title='🎥 Pinecone', page_title="👋 Pinecone!", slogan='The better way to search for films')
PINCONE_API_KEY = os.environ.get('PINCONE_API_KEY')
st.image('./images/pinecone.png')

# Streamlit tabs
tab1, tab2, tab3 = st.tabs(["General", "Create Index", "NLTK"])

with tab1:
    docs.pinecone_general()

with tab2:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    st.text(pc.list_indexes().names())
    st.text(pc.describe_index("d384"))

with tab3:
    pass