# rag_integration.py
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def ingest_notes(document_path):
    # Load document, split and index
    with open(document_path, 'r') as f:
        notes = f.read()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(notes)
    db = Chroma.from_documents(docs, OpenAIEmbeddings(openai_api_key="YOUR_KEY"))
    db.persist()
    return db
