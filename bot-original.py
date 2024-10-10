import os

from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.schema import Document

from CustomEmbeddings import CustomEmbeddings

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


query_embedding = "is this book about tao joga obraza?"
embeddings = genai.embed_content(model="models/embedding-001", content=query_embedding,
                                   task_type="retrieval_document",
                                 title="Embedding of single string")


# Perform the search in Pinecone, requesting top 5 most similar embeddings
#index = pc.Index(index_name=os.environ.get("INDEX_NAME"))
vector_list = embeddings['embedding']
custom_embeddings = CustomEmbeddings(vector_list)
vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=custom_embeddings)

query = "in which page and which ebook name talks about biorythm?"
result = vectorstore.similarity_search(query)

#print("Result: " + str(result))
# Initialize the model using LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",api_key=os.getenv("GEMINI_API_KEY"))

# Step 2: Load the QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Step 3: Prepare the Pinecone search results as LangChain Documents
# (assuming `pinecone_docs` is the list of documents returned from the Pinecone query)
documents = [
    Document(page_content=doc.page_content, metadata=doc.metadata) for doc in result
]


# Step 4: Ask the question (original query) and pass documents to the QA chain


# Step 5: Use the QA chain to generate the answer based on retrieved documents
response = qa_chain.run(input_documents=documents, question=query)

# Step 6: Output the response
print(response)

#docsearch = Pinecone.from_existing_index(index_name, embeddings)

#result = index.query(
 #   vector=query_embedding,
  #  top_k=5,  # Retrieve the top 5 similar embeddings
   # include_metadata=True  # To also return associated metadata
#)



