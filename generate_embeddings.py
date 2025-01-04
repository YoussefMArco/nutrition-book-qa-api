import os
import pickle
import time
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from typing import List
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK stopwords (run once)
nltk.download('stopwords')
nltk.download('wordnet')
# Specify the folder containing PDF documents
folder_path = r'/mnt/e/ML/projects/my_own_projects/nutrition/documents'

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess text
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # Remove special characters (keep numbers)
    text = re.sub(r'[^\w\s\d]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Function to process PDFs and extract metadata
def process_pdfs(folder_path: str) -> List[Document]:
    docs = []
    pdf_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_count += 1
            file_path = os.path.join(folder_path, filename)
            print(f"Processing PDF {pdf_count}: {filename}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            for page in pages:
                # Clean the text
                page.page_content = clean_text(page.page_content)
                # Add metadata (e.g., filename)
                page.metadata['source'] = filename
            docs.extend(pages)
    print(f"Total number of PDFs processed: {pdf_count}")
    return docs

# Function to split documents into chunks
def split_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Total number of chunks generated for embeddings: {len(chunks)}")
    return chunks

# Function to generate embeddings and create vectorstore
def create_vectorstore(docs: List[Document], persist_directory: str = "./chroma_db_nccn") -> Chroma:
    # Initialize the HuggingFace embeddings function
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Use 'cpu' if GPU is not available
    )

    # Create Chroma vectorstore and persist it
    print("Creating vectorstore...")
    start_time = time.time()
    vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
    end_time = time.time()
    print(f"Time taken to create vectorstore: {end_time - start_time} seconds")
    return vectorstore

# Main function
def main():
    # Check if processed documents already exist
    if os.path.exists("processed_docs.pkl"):
        print("Loading processed documents from file...")
        with open("processed_docs.pkl", "rb") as f:
            docs = pickle.load(f)
    else:
        print("Processing PDFs...")
        docs = process_pdfs(folder_path)
        print("Splitting documents into chunks...")
        docs = split_documents(docs)
        # Save processed documents to file
        with open("processed_docs.pkl", "wb") as f:
            pickle.dump(docs, f)

    # Create vectorstore
    vectorstore = create_vectorstore(docs)

    # Debugging message: Number of documents stored in vectorstore
    print(f"Number of documents stored in the vectorstore: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()