import streamlit as st, os, time
from streamlit_extras.stateful_button import button
import pandas as pd
import singlestoredb as s2
from utils import st_def

st_def.st_logo(title='🎥 SingleStore', page_title="👋 SingleStore!", slogan='Transactions + Analytics SQL + NoSQL Real-time RAG')
tab1, tab2, tab3 = st.tabs(["🔰MySQL, SingleStoreDB", "➡️SQLALchemy", "🪻SQL"])


with tab1:
    if button("Run? ", key="button1"):
        df = pd.read_csv("./data/singlestore/iris.csv", index_col=False)
        st.text(df.head())
        if button("Create Table? ", key="button12"):
            stmt = """INSERT INTO iris (sepal_length,sepal_width,petal_length,petal_width,species) VALUES (%s, %s, %s, %s, %s)"""
            with conn:
                conn.autocommit(True)
                with conn.cursor() as cur:
                    cur.execute("""CREATE TABLE IF NOT EXISTS iris (sepal_length FLOAT,sepal_width FLOAT,petal_length FLOAT,petal_width FLOAT,species VARCHAR(20))""")
                    cur.executemany(stmt, df)
                    cur.execute("""SELECT * FROM iris""")
                    rows = cur.fetchall()



with tab2:
    if button("Run 2?", key="button2"):
        pass
    # from sqlalchemy.ext.declarative import declarative_base
    # from sqlalchemy import create_engine
    #
    # Base = declarative_base()
    #
    #
    # class User(Base):
    #     __tablename__ = 'user'
    #     first_name = Column(VARCHAR(200))
    #     last_name = Column(VARCHAR(200), primary_key=True)
    #
    #
    # engine = create_engine(
    #     'mysql://svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com:3333')
    # engine.execute('USE database_92772')
    # Base.metadata.create_all(engine)

with tab3:
    if button("Run SQL? ", key="button3"):
        conn = s2.connect(host="svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com",
                          port=3333, user="aixpertlab", password=os.environ.get("SINGLE_STORE_DB_PASSWORD"),
                          database="database_92772", results_type="tuples")
        with conn:
            conn.autocommit(True)
            with conn.cursor() as cur:
                cur.execute("""SELECT * FROM iris""")
                rows = cur.fetchall()
        iris_df = pd.DataFrame(rows, columns=["sepal_length","sepal_width","petal_length","petal_width","species"])
        st.text(iris_df.head(11))

        import plotly.express as px
        from sklearn.decomposition import PCA

        X = iris_df[["sepal_length","sepal_width","petal_length","petal_width"]]

        pca = PCA(n_components=2)
        components = pca.fit_transform(X)

        pca_fig = px.scatter(
            components,            x=0,            y=1,            color=iris_df["species"]        )

        # pca_fig.show()
        st.plotly_chart(pca_fig)

        import matplotlib.pyplot as plt
        import seaborn as sns

        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.heatmap(
        #     iris_df.corr(),
        #     cmap="OrRd",
        #     annot=True,
        #     ax=ax
        # )
        # ax.set_title("Correlations")
        #
        # # Display the plot in Streamlit
        # st.title("Correlation Heatmap")
        # st.pyplot(fig)

        # Select the relevant features
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        # Create the correlation matrix
        corr_matrix = iris_df[features].corr()

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            cmap="OrRd",
            annot=True,
            ax=ax
        )
        ax.set_title("Correlations")

        # Display the plot in Streamlit
        st.title("Correlation Heatmap")
        st.pyplot(fig)