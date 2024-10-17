import streamlit as st
import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama

# Get Knowledge base from text file
document = SimpleDirectoryReader(input_files=['Rico Alzaga Q&A.txt']).load_data()

# Load embedding model 
embed_model = HuggingFaceEmbedding()

# Perform indexing and storing embeddings to disk
db = chromadb.PersistentClient(path="rico_qa_db")
chroma_collection = db.get_or_create_collection("rico_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    document,
    storage_context=storage_context,
    embed_model=embed_model
)

# Load embeddings from disk
chroma_collection = db.get_collection("rico_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

# Load LLM model to provide response from user  inquiries
def ask_rag(question):
    llm = Ollama(model="mistral", request_timeout=600.0)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(question)
    return response.response

# Provice UI functionality for the RAG system
st.title('ASK Rey Rico Alzaga')
question = st.text_input('Question')
if question.strip():
    answer = ask_rag(question)
    st.write(answer)
else:
    st.write('Ask anything about his personal and academic background, his looks, his hobbies and his professional life. :>')