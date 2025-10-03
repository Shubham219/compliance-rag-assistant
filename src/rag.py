# rag_example.py
#
# This script demonstrates a complete workflow for a simple RAG (Retrieval-Augmented Generation) system.
# It includes document chunking, creating embeddings with Sentence Transformers, building and saving a
# FAISS index, and performing a semantic search with a user query.

import os
import torch
import faiss
from sentence_transformers import SentenceTransformer

faiss_index_path = r'D:\PythonProjects\AML_Chat_Bot\faiss_index.bin'
model_path = r'D:\PythonProjects\AML_Chat_Bot\sentence_transformer_model'

# 2. Create the Sentence Transformer model and embeddings.
print("Loading Sentence Transformer model...")
# Using a lightweight but effective model for this example.
# 'all-MiniLM-L6-v2' maps sentences & paragraphs to a 384-dimensional dense vector space.
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

user_query = "Give me a brief summary on AML LAWS"

# -----------------------------------------------------------------------------------
# Now, let's simulate the next steps: loading the index and performing a search.
# -----------------------------------------------------------------------------------

# 5. Load the FAISS index and model from disk.
print("\nLoading FAISS index and model from disk for a new session...")
loaded_index = faiss.read_index(faiss_index_path)
loaded_model = SentenceTransformer(model_name)
print("Index and model loaded.")

# 6. User Query.
user_query = "What is the key difference between S-BERT and BERT for creating embeddings?"
print(f"\nUser query: '{user_query}'")

# 7. Convert the user query to an embedding.
query_embedding = loaded_model.encode([user_query], convert_to_tensor=True)
query_embedding_np = query_embedding.cpu().numpy()

# 8. Perform a vector similarity search.
# We want to find the top 3 most similar document chunks.
k = 3
print(f"Searching for the top {k} most similar documents...")
distances, indices = loaded_index.search(query_embedding_np, k)

# 9. Display the retrieved results.
print("\n--- Retrieved Documents ---")
for i in range(k):
    doc_index = indices[0][i]
    distance = distances[0][i]
    print(f"Rank {i + 1}:")
    print(f"  Document: '{documents[doc_index]}'")
    print(f"  Similarity Score (L2 distance): {distance:.4f}")
    # Note: L2 distance is smaller for more similar vectors.
    print("-" * 20)

print("Search complete.")

# Clean up created files for subsequent runs
if os.path.exists(faiss_index_path):
    os.remove(faiss_index_path)
if os.path.exists(model_path):
    # This will delete the directory created by model.save()
    import shutil
    shutil.rmtree(model_path)
