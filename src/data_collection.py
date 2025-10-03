import os
import requests
from typing import List

# Download FATF documents
#TODO: Add more URLs as needed

fatf_urls = [
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/FATF-Recommendations-2012.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/methodology/FATF-Methodology-2013.pdf"
]


def download_documents(urls: List[str], download_dir: str):
    """
    Downloads documents from a list of URLs and saves them to a directory.

    Args:
        urls (List[str]): A list of URLs to the documents (e.g., PDFs).
        download_dir (str): The directory where the documents will be saved.
    """
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")

    for url in urls:
        file_name = url.split('/')[-1]
        file_path = os.path.join(download_dir, file_name)

        # Check if the file already exists to avoid re-downloading
        if os.path.exists(file_path):
            print(f"File already exists, skipping: {file_name}")
            continue

        print(f"Downloading {file_name}...")
        try:
            # Send a GET request to the URL with a timeout
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Write the content to a file in binary mode
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded and saved: {file_name}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name}: {e}")
            
if __name__ == "__main__":

    # The directory where the documents will be stored
    output_directory = "documents/fatf"
    
    print("Starting the data collection process...")
    download_documents(fatf_urls, output_directory)
    print("\nData collection process completed.")
