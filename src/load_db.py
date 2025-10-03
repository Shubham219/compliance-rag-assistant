# step_3_rag_generation.py
#
# This script loads the FAISS index and documents, performs a semantic search,
# and then uses the retrieved documents as context to generate a response
# using a Large Language Model (LLM) via an API call.

import os
import torch
import faiss
import requests
import json
from sentence_transformers import SentenceTransformer

# Define the path where the FAISS index file and the documents file are located.
faiss_index_path = 'faiss_index.bin'
documents_file_path = 'documents\chunked_text\AML_CFT_Measures_and_Financial_Inclusion_2013.pdf.coredownload_chunks.txt'

# In a real-world scenario, you would have saved and reloaded the model,
# but for this example, we will just instantiate it again.
model_name = 'all-MiniLM-L6-v2'

# --- LLM API Configuration ---
# IMPORTANT: Replace with your actual API key and endpoint
# This is a placeholder for demonstration purposes.
API_KEY = "YOUR_API_KEY"
LLM_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + API_KEY

def call_llm_api(prompt):
    """
    Sends a prompt to the LLM and returns the generated text.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # Parse the JSON response
        result = response.json()
        
        # Extract the generated text from the response
        generated_text = result['candidates'][0]['content']['parts'][0]['text']
        return generated_text

    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        print("Raw response:", response.text)
        return None


# --- Main Script ---
print("Starting RAG generation process...")

try:
    # Check if the FAISS index and documents files exist before proceeding.
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index file not found: {faiss_index_path}")
    if not os.path.exists(documents_file_path):
        raise FileNotFoundError(f"Documents file not found: {documents_file_path}")

    # Load the FAISS index from the binary file.
    loaded_index = faiss.read_index(faiss_index_path)
    print(f"FAISS index successfully loaded. It contains {loaded_index.ntotal} vectors.")

    # Now, load the original documents from the text file.
    with open(documents_file_path, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    print(f"{len(documents)} documents loaded.")

    # Load the Sentence Transformer model for query embedding.
    print("\nLoading Sentence Transformer model for query embedding...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    # 1. Get the user's query.
    user_query = "Explain me in the brief the AML LAWS"
    print(f"\nUser query: '{user_query}'")

    # 2. Convert the user query to an embedding.
    query_embedding = model.encode([user_query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()

    # 3. Perform a vector similarity search.
    k = 3
    print(f"Searching for the top {k} most similar documents...")
    distances, indices = loaded_index.search(query_embedding_np, k)

    # 4. Prepare context for the LLM.
    retrieved_context = ""
    for i in range(k):
        doc_index = indices[0][i]
        retrieved_context += documents[doc_index] + "\n---\n"
    print("\nRetrieved context for LLM:", retrieved_context)

    # 5. Build the prompt for the LLM.
    # The prompt instructs the LLM to use only the provided context.
    llm_prompt = f"""
    Answer the following question based ONLY on the context provided.
    If the answer cannot be found in the context, respond with "I cannot answer this question based on the provided information."

    Context:
    {retrieved_context}

    Question: {user_query}
    """
    
    # 6. Call the LLM API to get the final answer.
    print("\nSending context to LLM for final generation...")
    final_answer = call_llm_api(llm_prompt)

    # 7. Print the final result.
    print("\n--- Final Generated Answer ---")
    if final_answer:
        print(final_answer)
    else:
        print("Failed to get an answer from the LLM.")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure you have created the required FAISS and documents files first.")
