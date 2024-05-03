import streamlit as st, time, platform
from streamlit import logger
import sqlite3, chromadb, pathlib
from utils import st_def, ut_vector

st_def.st_logo(title="Welcome 👋 to Chroma!", page_title="Summary", )
st.write(platform.processor())
st.write(logger.get_logger("SMI_APP"))
#---------------------------------------------------------------------------------------------------------
def chroma_collection(name):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name=name)
    return collection


tab1, tab2, tab3 = st.tabs(["Prototype", "General", "NLTK"])
with tab1:
    st.image('./images/chroma.png')
    st.info('no persistence, just in memory')
    collection = chroma_collection(name="collection1_1")
    collection.add(
        documents=["steak", "python", "tiktok", "safety", "health", "environment"],
        metadatas=[{"source": "food"}, {"source": "progamming language"}, {"source": "social media"},
                   {"source": "government"}, {"source": "body"}, {"source": "living condition"}],
        ids=["id1", "id2", "id3", "id4", "id5", "id6"])
    qa = st.text_input('🌐steak python tiktok safety health environment')
    if qa:
        results = collection.query(query_texts=[qa], n_results=1)
        st.write(results)
#------------------------------------------------------------------------

with tab2:
    st.image('./images/chroma2.png')
    st.markdown("Chroma是一款开源的嵌入式数据库。""")