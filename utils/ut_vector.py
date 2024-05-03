import sys, os, platform

if not platform.processor()[:7]=="Intel64":
  __import__('pysqlite3'); 
  sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
  print(f'server-------------------  {platform.processor()}')
else:
  print(f'local-------------------  {platform.processor()}')


import sqlite3, chromadb, pathlib
from chromadb.utils import embedding_functions
from more_itertools import batched



def car_review_collection(
  chroma_path: pathlib.Path,
  collection_name: str,
  embedding_func_name: str,
  ids: list[str],
  documents: list[str],
  metadatas: list[dict],distance_func_name: str = "cosine",):
  
  """Create a ChromaDB collection"""

  chroma_client = chromadb.PersistentClient(chroma_path)
  # chroma_client = chromadb.Client()
  embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_func_name)

  collection = chroma_client.get_or_create_collection(
      name=collection_name,
      embedding_function=embedding_func,
      metadata={"hnsw:space": distance_func_name},)

  document_indices = list(range(len(documents)))

  for batch in batched(document_indices, 166):
      start_idx = batch[0]
      end_idx = batch[-1]

      collection.add(
          ids=ids[start_idx:end_idx],
          documents=documents[start_idx:end_idx],
          metadatas=metadatas[start_idx:end_idx],
      )
  return collection