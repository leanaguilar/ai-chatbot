import os
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.schema import Document

from CustomEmbeddings import CustomEmbeddings

load_dotenv()

class ChatBot:
    def __init__(self):
        # Initialize Pinecone, Google Generative AI, and load environment variables
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.index_name = os.environ.get("INDEX_NAME")

        # Setup Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Setup Google Generative AI (Gemini)
        genai.configure(api_key=self.gemini_api_key)

        # Initialize the LLM for QA chain
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=self.gemini_api_key)

    def get_response(self, message):
        """
        Receives a message (query) and returns a response based on embeddings and similarity search in Pinecone.
        If no documents are found, it queries the Google Gemini API directly for information.
        """
        try:
            # Step 1: Create embedding for the incoming query message
            embeddings = genai.embed_content(
                model="models/embedding-001",
                content=message,
                task_type="retrieval_document",
                title="Embedding of user query"
            )

            # Step 2: Extract vector list from embedding
            vector_list = embeddings['embedding']
            custom_embeddings = CustomEmbeddings(vector_list)


            # Step 3: Perform similarity search in Pinecone
            vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=custom_embeddings)
            result = vectorstore.similarity_search(message)
            #print(result)

            # Set a similarity threshold (adjust this value based on your requirements)
            similarity_threshold = 0.7  # Example threshold (0.0 to 1.0)

            # Step 4: Convert Pinecone search results to LangChain Document objects
            documents = [
                Document(page_content=doc.page_content, metadata=doc.metadata) for doc in result if doc.page_content
            ]

            # Check if we have found any relevant documents
            if not documents:
                # If no documents were found, query the LLM directly for more information
                return self.query_llm(message)

            # Step 5: Load the QA chain using LangChain
            qa_chain = load_qa_chain(self.llm, chain_type="stuff")

            # Step 6: Run the QA chain with the retrieved documents and user query
            response = qa_chain.run(input_documents=documents, question=message)

            return response

        except Exception as e:
            return f"Error: {str(e)}"

    def query_llm(self, query):
        """
        Directly query the Google Gemini LLM for information when no documents are found.
        """
        try:
            # Construct the message in a way that the LLM expects
            response = self.llm.generate([{"role": "user", "content": query}])

            # Extract the text from the response
            if response and isinstance(response, list) and len(response) > 0:
                return response[0].get("content", "No response content found.")
            else:
                return "No response received from LLM."
        except Exception as e:
            return f"Error while querying LLM: {str(e)}"


#chatbot = ChatBot()
#print(chatbot.get_response("Please I want a precise and relevant reponse. Who is savina atai?"))
