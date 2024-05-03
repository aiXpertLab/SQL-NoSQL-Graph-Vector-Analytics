import streamlit as st, os, time, csv, pickle, requests, json
from streamlit_extras.stateful_button import button
from utils import st_def, tmdb, mypinecone, docs

st_def.st_logo(title='🎥 Weaviate', page_title="👋 Weaviate!", slogan='Weaviate (we-vee-eight) is an open source, AI-native vector database. ')
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🔰Create Cluster", "➡️Define Collection", "🪻Add Objects", "Embedding🍍", "Vector🎍", "Retrieval➡️", "Q&A➡️", "Evaluation🏅"])
with tab1:
    import weaviate
    import weaviate.classes as wvc

    openai_key = os.environ.get("OPENAI_API_KEY", "")

    # Setting up client
    client = weaviate.Client(
        url=os.getenv("WCS_URL"),
        auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY")),  # Replace with your WCS key
        additional_headers={'X-OpenAI-Api-key': os.getenv("OPENAI_API_KEY")}  # Replace with your vectorizer API key
    )
    st.info(f"client.is_ready()?    {client.is_ready()}")


with tab2:
    schema = client.schema.get()
    collections = [d["class"] for d in schema["classes"]]
    for collection in collections:
        st.code(collection)

    if "Questions" in collections:
        st.info(f"Collection 'Questions' exists.")
    else:
        questions = client.schema.create(
            name="Question2",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),            # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
            generative_config=wvc.config.Configure.Generative.openai()            # Ensure the `generative-openai` module is used for generative queries
        )

# with tab3:
#     resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
#     data = json.loads(resp.text)  # Load data
#
#     question_objs = list()
#     for i, d in enumerate(data):
#         question_objs.append({
#             "answer": d["Answer"],
#             "question": d["Question"],
#             "category": d["Category"],
#         })
#
#     questions = client.collections.get("Question")
#     questions.data.insert_many(question_objs)
#     st.write(questions)
#     st.write(question_objs)