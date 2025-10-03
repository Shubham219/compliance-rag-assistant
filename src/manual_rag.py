from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embedder(chunk):
    """
    embed corpus of documents
    """
    tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        model_output = model(**tokens)

    embeddings = model_output.last_hidden_state[:, 0, :]
    embed = embeddings[0].numpy()
    return embed

# Write a function to read a text file and return its content as list of text chunks of length 512 with overlap of 100
def read_file_in_chunks(file_path: str, chunk_size: int = 512, overlap: int = 100):
    with open(file_path, "r") as file:
        text = file.read()

    # Split text into chunks with overlap
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])

    return chunks

chunks = read_file_in_chunks("regulatory_documents/fatf_cleaned_2.txt")
print(chunks)