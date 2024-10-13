import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import pinecone
from langchain.embeddings.base import Embeddings
from langchain.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document  # Import Document class
from CustomEmbeddings import CustomEmbeddings

# Load environment variables
load_dotenv()

# Initialize Google Gemini for embeddings
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize Pinecone
pc = Pinecone(pinecone_api_key=os.environ.get("PINECONE_API_KEY"))


# Function to load PDF and extract text
def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a single string.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Load the PDF file
pdf_file_name = "face.pdf"
pdf_text = load_pdf(file_path=pdf_file_name)


# Function to split text into chunks based on paragraphs
def split_text(text: str):
    """
    Splits a text string into a list of non-empty substrings based on paragraphs.
    """
    split_text = re.split('\n\n', text)  # Split based on double new lines (paragraphs)
    return [i for i in split_text if i.strip()]  # Return only non-empty chunks


# Split the PDF text into chunks
chunked_text = split_text(text=pdf_text)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

# Load and split PDF into smaller chunks for embedding
loader = PyPDFLoader(pdf_file_name)
documents = loader.load()

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# Instantiate the CustomEmbeddings class
custom_embeddings = CustomEmbeddings()


# Function to store the chunks in Pinecone using the embeddings model
def store_in_pinecone(chunks, embedding_model, file_name):
    """
    Processes chunks of text and stores them in Pinecone using the provided embedding model.
    """
    index_name = os.environ.get("INDEX_NAME")

    # Prepare documents and generate embeddings
    # Create Document objects from chunks
    documents = [
        Document(
            page_content=chunk.page_content,
            metadata={
                "page_number": chunk.metadata.get('page', -1),  # Default to -1 if not available
                "ebook_name": file_name  # Store the ebook name
            }
        ) for chunk in chunks
    ]

    # Generate embeddings for each chunk
    embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])  # Generate embeddings

    # Store using PineconeVectorStore
    PineconeVectorStore.from_documents(
        documents=documents,  # Pass the list of Document objects
        embedding=custom_embeddings,  # Pass your custom embedding model
        index_name=index_name
    )


# Call the function to store the chunks into Pinecone
store_in_pinecone(chunks, custom_embeddings, pdf_file_name)

index_name = os.environ.get("INDEX_NAME")
print(f"Documents stored successfully in Pinecone index: {index_name}")
