from pypdf import PdfReader
from langchain.document_loaders import PyPDFLoader
import os
import json
from pypdf import PdfReader
import os
import re

def is_index_or_table_page(text):
    """
    Heuristically determines if a page is an index, table of contents, or list of figures.
    Returns True if so, False otherwise.
    """
    # Lowercase for easier matching
    lower = text.lower()
    # Common keywords for such pages
    keywords = [
        "table of contents", "contents", "index", "list of tables", "list of figures", "figures", "tables"
    ]
    # If the page is short and contains one of the keywords, likely an index/table page
    if any(kw in lower for kw in keywords) and len(text.split()) < 100:
        return True
    # If the page is mostly numbers or page references
    if re.match(r"^\s*\d+(\s+\d+)*\s*$", text.replace('\n', ' ')):
        return True
    return False

def clean_text(text):
    """
    Cleans the input text by removing extra whitespace and unwanted characters.
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove unwanted characters (e.g., non-ASCII)
    text = ''.join(c for c in text if ord(c) < 128)
    # Remove multiple newlines
    text = '\n'.join([line for line in text.splitlines() if line.strip() != ''])
    return text

def remove_header_footer(page_text):
    """
    Removes header and footer from a page's text.
    Assumes header is the first line and footer is the last line of the page.
    You can adjust this logic for your specific documents.
    """
    lines = page_text.splitlines()
    # Remove empty lines at start/end
    lines = [line for line in lines if line.strip()]
    if len(lines) > 3:
        # Remove first and last line (header/footer)
        lines = lines[3:]
    return "\n".join(lines)

def pdf_folder_to_clean_texts(pdf_folder, output_path):
    """
    Reads all PDFs from a folder, extracts and cleans text, removes index/table pages and headers/footers, and saves to a text file.
    """
    all_texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            print(f"Processing {filename}...")
            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                if not raw_text.strip():
                    continue
                # Remove header and footer
                raw_text = remove_header_footer(raw_text)
                # print 2 pages and exit for inspection
                if i < 9:
                    print(f"--- Page {i+1} of {filename} ---")
                    print(raw_text)
                    print("------------------------------")
                if is_index_or_table_page(raw_text):
                    print(f"Skipping page {i+1} (index/table/figure): {filename}")
                    continue
                cleaned = clean_text(raw_text)
                if cleaned:
                    all_texts.append(cleaned)
    # Save all cleaned texts to output file
    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n\n")
    print(f"Saved cleaned text to {output_path}")

# Example usage:
if __name__ == "__main__":
    fatf_dir = "documents/fatf"
    output_txt = "documents/fatf_cleaned_2.txt"
    if not os.path.exists(fatf_dir):
        print(f"Directory not found: {fatf_dir}")
    else:
        pdf_folder_to_clean_texts(fatf_dir, output_txt)


# def process_documents(documents_folder):
#     processed_docs = []
    
#     for filename in os.listdir(documents_folder):
#         if filename.endswith('.pdf'):
#             # Load document
#             loader = PyPDFLoader(f"{documents_folder}/{filename}")
#             docs = loader.load()
            
#             # Add metadata
#             for doc in docs:
#                 doc.metadata['source'] = filename
#                 doc.metadata['document_type'] = 'aml_policy'
            
#             processed_docs.extend(docs)
    
#     return processed_docs

# def clean_text(text):
#     """
#     Cleans the input text by removing extra whitespace and unwanted characters.
    
#     Args:
#         text (str): The input text to clean.

#     Returns:
#         str: The cleaned text.
#     """
#     # Remove extra whitespace
#     text = " ".join(text.split())
#     # Remove unwanted characters (e.g., non-ASCII)
#     text = ''.join(c for c in text if ord(c) < 128)
#     # Remove multiple newlines
#     text = '\n'.join([line for line in text.splitlines() if line.strip() != ''])
#     return text

# if __name__ == "__main__":
#     # Directory containing FATF PDFs
#     fatf_dir = "documents/fatf"
#     if not os.path.exists(fatf_dir):
#         print(f"Directory not found: {fatf_dir}")
#     else:
#         print(f"Processing PDF files in: {fatf_dir}")
#         # Process and load documents
#         docs = process_documents(fatf_dir)
#         print(f"Loaded {len(docs)} documents.")
        
#         # Clean the text of each document
#         for doc in docs:
#             doc.page_content = clean_text(doc.page_content)

#         # Save processed documents to a JSONL file
#         output_path = "documents/fatf_processed.jsonl"
#         with open(output_path, "w", encoding="utf-8") as f:
#             for doc in docs:
#                 f.write(json.dumps(doc.page_content, ensure_ascii=False) + "\n")  

#         # Write a unit test for clean_text function
#         def test_clean_text():
#             raw_text = "This is a test.\n\nThis is only a test!   \n\n\nExtra spaces and non-ASCII: ñ, ü, é."
#             expected_cleaned_text = "This is a test. This is only a test! Extra spaces and non-ASCII: , , ."
#             assert clean_text(raw_text) == expected_cleaned_text
#             print("clean_text function passed the test.")
#         test_clean_text()