from langchain.embeddings.base import Embeddings
import google.generativeai as genai

class CustomEmbeddings(Embeddings):

    def __init__(self, vectors=None):
        # If no vectors are provided, initialize with an empty list
        self.vectors = vectors if vectors is not None else []

    def embed_documents(self, texts):
        """
        Given a list of texts, return their embeddings.
        """
        embeddings = []
        for text in texts:
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
                title="Embedding of document chunk"
            )
            embeddings.append(embedding['embedding'])  # Extract the embedding vector
        return embeddings

    def embed_query(self, text):
        """
        Given a query string, return its embedding.
        """
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Embedding of query"
        )
        return embedding['embedding']  # Extract the embedding vector