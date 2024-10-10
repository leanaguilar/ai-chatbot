from langchain.embeddings.base import Embeddings
import google.generativeai as genai

class CustomEmbeddings(Embeddings):
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_documents(self, documents):
        # This method will return the precomputed embeddings (vectors) in the same order as documents
        return self.vectors

    def embed_query(self, query: str):
        """
        Embed the query using the same embedding model.
        """
        # Call genai.embed_content() to get the query embedding
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=[query],  # Pass the query in a list to embed_content
            task_type="retrieval_document",
            title="Embedding of query string"
        )

        # Extract and return the embedding vector
        return query_embedding['embedding'][0]  # Extract the first embedding

