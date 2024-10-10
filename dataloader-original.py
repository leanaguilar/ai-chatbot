import re
from PyPDF2 import PdfReader
from langchain import text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pypdf
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
import pinecone

from CustomEmbeddings import CustomEmbeddings

def load_pdf(file_path):
    """
    Reads the text content from a PDF file and returns it as a single string.

    Parameters:
    - file_path (str): The file path to the PDF file.

    Returns:
    - str: The concatenated text content of all pages in the PDF.
    """
    # Logic to read pdf
    reader = PdfReader(file_path)

    # Loop over each page and store it in a variable
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text
pdf_text = load_pdf(file_path="libro_sabina_mini.pdf")

def split_text(text: str):
    """
    Splits a text string into a list of non-empty substrings based on the specified pattern.
    The "\n \n" pattern will split the document para by para
    Parameters:
    - text (str): The input text to be split.

    Returns:
    - List[str]: A list containing non-empty substrings obtained by splitting the input text.

    """
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

chunked_text = split_text(text=pdf_text)

#print(chunked_text)

#exit(0)

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
#cargo PDF
loader = PyPDFLoader("libro_sabina_mini.pdf")
documents = loader.load()

#uno el contenido de pagina en un solo texto
text = ""
for page in documents:
    text += page.page_content

#divido todo el texto en chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts = text_splitter.split_text(text)

#text = documents[1].page_content
#print(texts)

#text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
#texts = text_splitter.split_text(documents)
#print(f"created {len(texts)} chunks")

#textsDoc = text_splitter.split_documents(documents)

#def chunk_data(docs, chunk_size=800, chunk_overlap=50):
 #   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  #  doc = text_splitter.split_documents(docs)
   # return docs
#documents2 = chunk_data(docs=documents)

#divido documento en partes
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100 )
chunks = text_splitter.split_documents(documents)
# Call Chunks
print(len(chunks))
#print(chunks[20].page_content)

#print(embeddings['embedding'])

#pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
#index_name = "pdf-sabina"
#index = pc.Index(index_name)
#vector_store = PineconeVectorStore(index=index, embedding=embeddings)

#print(texts)
#for chunk in texts:
 #   print(chunk)

#hay que hacer embedding y store por cada chunk?
embeddings = genai.embed_content(model="models/embedding-001", content=texts,
                                   task_type="retrieval_document",
                                 title="Embedding of single string")
#print(embeddings)
# Step 1: Extract vectors from EmbeddingDict
vector_list = embeddings['embedding']

# Step 2: Create an instance of CustomEmbeddings using the extracted vectors
custom_embeddings = CustomEmbeddings(vector_list)

PineconeVectorStore.from_documents(chunks,custom_embeddings, index_name=os.environ.get("INDEX_NAME"))


