import os
import uuid
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv(".env")

# Set the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the Qdrant collection name
qdrant_collection = 'Ashara'


def process_pdfs(directory, max_chunk_size=4096):
    """
    Processes all PDF files in the specified directory, extracting text from them.

    Args:
        directory (str): Directory containing PDF files.
        max_chunk_size (int): Maximum size of text chunks to be extracted.

    Returns:
        list: List of text extracted from the PDFs.
    """
    documents = []

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            reader = PdfReader(pdf_path)
            
            for page in reader.pages:
                documents.append(page.extract_text())
    
    return documents


def generate_embeddings(text, api_key):
    """
    Generates embeddings for the provided text using OpenAI.

    Args:
        text (str): Text for which embeddings are generated.
        api_key (str): OpenAI API key for authentication.

    Returns:
        list: Embedding vector.
    """
    embed = OpenAIEmbeddings(openai_api_key=api_key)
    embeddings = embed.embed_query(text)
    return embeddings


def save_embeddings_to_qdrant(documents, client):
    """
    Saves document embeddings to the Qdrant collection.

    Args:
        documents (list): List of document texts.
        client (QdrantClient): Qdrant client instance.

    Returns:
        QdrantClient: The client instance after upserting embeddings.
    """
    points = []

    for doc in documents:
        try:
            embeddings_vector = generate_embeddings(doc, openai_api_key)
        except Exception as e:
            print(f"Error generating embeddings for document '{doc[:100]}': {e}")
            continue  # Skip document on error

        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embeddings_vector,
            payload={"text": doc}
        ))

    client.upsert(collection_name=qdrant_collection, points=points)
    return client


def create_qdrant_collection(force_recreate=False):
    """
    Creates a new Qdrant collection.

    Args:
        force_recreate (bool): If true, recreate the collection if it exists.

    Returns:
        QdrantClient: Initialized Qdrant client.
    """
    client = QdrantClient(":memory:")
    if force_recreate:
        client.recreate_collection(
            collection_name=qdrant_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    return client


def main():
    """
    Main function to process PDF, generate embeddings, and save them to Qdrant.
    """
    client = create_qdrant_collection()

    # Set the path where PDFs are located
    pdf_directory = r"D:\ashara_lamda_function\documents\\"
    
    # Process PDFs and extract document texts
    documents = process_pdfs(pdf_directory)

    if documents:
        # Save the document embeddings to Qdrant
        db = save_embeddings_to_qdrant(documents, client)
        print('Embeddings generated and saved to Qdrant (local storage)')

        # Save the Qdrant data to a pickle file
        with open(r"D:\ashara_lamda_function\ashara_chat.pkl", "wb") as f:
            pickle.dump(db, f)
    else:
        print("No documents created from PDF. Please check the file or adjust processing logic.")


if __name__ == "__main__":
    main()
