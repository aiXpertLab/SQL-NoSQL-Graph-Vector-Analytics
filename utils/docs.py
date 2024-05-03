import streamlit as st


def pinecone_general():
    st.markdown("""
        #### Pinecone index
        In Pinecone, an index represents the top-level organizational structure for vector data. 
        It is responsible for receiving and storing vectors, handling queries on the stored vectors, 
        and performing various vector operations within its data. 

        Each index operates on one or more pods to ensure efficient functionality.

        #### Pods
    
        Pods are pre-configured hardware units that host Pinecone services. Each index in Pinecone operates on one or multiple pods, and having additional pods generally results in increased storage capacity, reduced latency, and improved throughput. Furthermore, users have the flexibility to create pods of varying sizes to meet specific requirements.
    
        #### Distance Metrics
        Distance Metrics are mathematical techniques used to assess the similarity between two vectors within a vector space. In vector databases, these measures play a crucial role in comparing the stored vectors with a given query vector to identify the most similar ones.
    
        You can choose from different metrics when creating a vector index:
    
        - Cosine similarity. This measure evaluates the cosine of the angle between two vectors in the vector space. Its scale ranges from -1 to 1, where 1 signifies identical vectors, 0 represents orthogonal vectors, and -1 indicates diametrically opposed vectors.
        - Euclidean distance. This measure calculates the straight-line distance between two vectors in the vector space. It ranges from 0 to infinity, where 0 denotes identical vectors, and larger values indicate increasingly dissimilar vectors.
        - Dot product. This measure computes the product of the magnitudes of two vectors and the cosine of the angle between them. Its scale spans from -∞ to ∞, where positive values indicate vectors pointing in the same direction, 0 represents orthogonal vectors, and negative values signify vectors pointing in opposite directions.
        """)
    st.subheader('By utilizing the Pinecone schema, we can incorporate custom dictionaries, like metadata, while creating vectors, providing us with the flexibility to perform searches based on arrays of tokens or publication year of the sentence. Take a look at the JSON object utilized to create a vector for a particular sentence.')
    st.code("""

{
     'id': 5226,
     'vector': [-0.006197051145136356, 
        0.004149597603827715, 
        0.007294267416000366,
       -0.02623748779296875,
       -0.027276596054434776...],
     'metadata': { 
           'is_duplicate': False, 
           'sentence_length': 51, 
           'tokens': ['python','programming','language'],
           'published_year': 2020
           }
}
    
    """)

    st.write("""
    
    The Milvus database schema does not accommodate dictionaries or arrays of strings; however, you can add custom string or integer properties or arrays of integers. Please refer to the supported data types in Milvus. The following JSON object illustrates how a vector is created in Milvus.

{
  'id': 5226,
  'vector': [-0.006197051145136356, 
        0.004149597603827715, 
        0.007294267416000366,
       -0.02623748779296875,
       -0.027276596054434776...],  
   'is_duplicate': False, 
   'sentence_length': 51, 
   'published_year': 2020
}
Summary:

Major difference between pinecone and milvus are their infrastructure and database schema. If you are looking for a simple and easy-to-use vector database with managed services and automated indexing, Pinecone might be the better choice. However, if you require in-house infrastructure, open-source, more flexibility in terms of indexing algorithms and need to customize your infrastructure to meet your specific needs, Milvus may be the better option.
    """)
