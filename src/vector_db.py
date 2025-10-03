import os
import faiss
import numpy as np
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",     # Pre-trained model for text understanding
    model_kwargs={'device': 'cpu'}      # Run on CPU (change to 'cuda' for GPU)
)

def load_jsonl_chunks(jsonl_path):
    """
    Loads chunks from a JSONL file.
    Each line should be a JSON object with 'content' and 'metadata' fields.
    Returns a list of dicts.
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            # Ensure keys are correct for downstream usage
            # Accept both 'content' or 'page_content'
            if 'content' in chunk:
                chunk['page_content'] = chunk['content']
            chunks.append(chunk)
    return chunks

def create_vector_database(processed_chunks):
    # Extract just the text content from chunks
    texts = [chunk.get('page_content', chunk.get('content', '')) for chunk in processed_chunks]
    # Extract metadata for each chunk
    metadatas = [chunk.get('metadata', {}) for chunk in processed_chunks]
    print(f"Creating vectors for {len(texts)} document chunks...")
    vector_db = FAISS.from_texts(
        texts=texts,           # Our document chunks
        embedding=embeddings,  # The model that converts text to numbers
        metadatas=metadatas    # Document source info, page numbers, etc.
    )
    return vector_db

def save_vector_database(vector_db, save_path="aml_compliance_db"):
    vector_db.save_local(save_path)
    print(f"Vector database saved to {save_path}")

def load_vector_database(save_path="aml_compliance_db"):
    vector_db = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading FAISS
    )
    print(f"Vector database loaded from {save_path}")
    return vector_db

def test_similarity_search(vector_db, query, top_k=3):
    print(f"Searching for: '{query}'")
    results = vector_db.similarity_search(
        query=query,    # User's question
        k=top_k         # Number of results to return
    )
    # Display results
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")  # First 200 characters
    return results

if __name__ == "__main__":
    # Path to the JSONL file containing processed document chunks
    jsonl_file_path = "documents/fatf_chunks.jsonl"
    if not os.path.exists(jsonl_file_path):
        print(f"File not found: {jsonl_file_path}")
    else:
        print(f"Loading document chunks from: {jsonl_file_path}")
        # Load processed chunks
        chunks = load_jsonl_chunks(jsonl_file_path)
        print(f"Loaded {len(chunks)} chunks from JSONL.")

        # Create vector database
        vector_db = create_vector_database(chunks)
        print("Vector database created successfully.")

        # Save vector database locally
        save_vector_database(vector_db, save_path="aml_compliance_db")

        # Example: Load vector database back
        loaded_db = load_vector_database(save_path="aml_compliance_db")

        # Test similarity search
        sample_query = "What are the FATF recommendations?"
        test_similarity_search(loaded_db, sample_query)
