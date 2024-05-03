import streamlit as st, os, time, csv, pickle
from streamlit_extras.stateful_button import button
from utils import st_def, tmdb, mypinecone
st_def.st_logo(title='🎥 Film Search', page_title="👋 Pinecone!", slogan='The better way to search for films')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TMDB_API_KEY    = os.environ.get('TMDB_API_KEY')
OPENAI_API_KEY  = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY= os.environ.get('PINECONE_API_KEY')
CSV_HEADER = ['Title', 'Runtime (minutes)', 'Language', 'Overview','Release Year', 'Genre', 'Keywords','Actors', 'Directors', 'Stream', 'Buy', 'Rent', 'Production Companies']
YEAR = '2024'
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# streamlit run app.py
if __name__ == "__main__":
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🔰Saving ids", "➡️CSV", "🪻To Pinecon", "Embedding🍍", "Vector🎍", "Retrieval➡️", "Q&A➡️", "Evaluation🏅"])
    with tab1:
        if button("Retrieve and Save", key="button1"):
            ids = tmdb.get_id_list(TMDB_API_KEY, YEAR)
            st.write(type(ids))
            with open('ids.pickle','wb') as f: pickle.dump(ids, f)

    with tab2:
        if not os.path.exists('ids.pickle'):st.info('no ids pickle existed.')
        else:
            with open('ids.pickle', 'rb') as f:ids = pickle.load(f)
        
        if button("Write to CSV", key="button2"):tmdb.write_ids_csv(ids, YEAR, CSV_HEADER, TMDB_API_KEY)
        st.success('in data/csv')
        
    with tab3:
        docs, splits = mypinecone.load_csv()
        st.write(len(docs))
        st.write(len(splits))
        st.text(docs[9679].page_content)
        