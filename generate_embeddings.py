import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Specify the folder containing PDF documents
folder_path = 'D:\\Huawei NTI Ai Training\\Projecttt\\Documents'

# Initialize an empty list to hold all documents
docs = []

# Counter for the number of PDFs processed
pdf_count = 0

# Loop through all files in the folder and load PDFs
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        pdf_count += 1
        file_path = os.path.join(folder_path, filename)
        print(f"Processing PDF {pdf_count}: {filename}")
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

# Debugging message: Number of PDFs processed
print(f"Total number of PDFs processed: {pdf_count}")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# Debugging message: Number of chunks generated
print(f"Total number of chunks generated for embeddings: {len(docs)}")

# Initialize the HuggingFace embeddings function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create Chroma vectorstore and persist it
vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn")

# Debugging message: Number of documents stored in vectorstore
print(f"Number of documents stored in the vectorstore: {vectorstore._collection.count()}")