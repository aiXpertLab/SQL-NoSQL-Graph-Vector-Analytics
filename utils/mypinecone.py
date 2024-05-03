from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_csv():
    loader = DirectoryLoader(
        path="./data",
        glob="*.csv",
        loader_cls=CSVLoader,
        show_progress=True)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return docs, splits


