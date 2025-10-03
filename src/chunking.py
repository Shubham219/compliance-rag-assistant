import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_processor import process_documents, clean_text
import json

# Legal document chunking preserving structure
splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,           # Keep chunks manageable
    chunk_overlap=200,         # Preserve context between chunks
    separators=["\n\n", "\n", ".", " "],  # Respect document structure
    keep_separator=True
)

def chunk_legal_documents(docs):
    chunks = splitter.split_documents(docs)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)
    
    return chunks


# get the documents from the document processor and chunk them
if __name__ == "__main__":
    fatf_dir = "documents/fatf"
    if not os.path.exists(fatf_dir):
        print(f"Directory not found: {fatf_dir}")
    else:
        print(f"Processing and chunking PDF files in: {fatf_dir}")
        docs = process_documents(fatf_dir)
        print(f"Loaded {len(docs)} documents.")
        
        # Clean the text of each document
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        chunks = chunk_legal_documents(docs)
        print(f"Created {len(chunks)} chunks.")

        # Save chunks to a JSONL file
        output_path = "documents/fatf_chunks.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps({
                    "chunk_id": chunk.metadata['chunk_id'],
                    "chunk_size": chunk.metadata['chunk_size'],
                    "source": chunk.metadata['source'],
                    "document_type": chunk.metadata['document_type'],
                    "page_content": chunk.page_content
                }, ensure_ascii=False) + "\n")
        print(f"Chunks saved to {output_path}")