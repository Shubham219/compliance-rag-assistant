import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

HOME_URL = "https://rulebook.centralbank.ae/"
TARGET_URL = "https://rulebook.centralbank.ae/en/rulebook/amlcft"

# Browser-like headers
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36')
}

def get_links(session, url):
    """Get all absolute links from a page."""
    resp = session.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        link_url = urljoin(url, a['href'])
        links.add(link_url)
    return links

def get_page_text(session, url):
    """Fetch and return visible text from a page."""
    resp = session.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Extract visible text, can be refined as needed
    texts = soup.stripped_strings
    return "\n".join(texts)

def main():
    with requests.Session() as session:
        session.headers.update(HEADERS)
        # Establish session by visiting home for cookies
        session.get(HOME_URL)
        
        # Extract all links from target AML/CFT page
        links = get_links(session, TARGET_URL)
        print(f"Found {len(links)} links on target page.\n")

        # Directory to save the text files
        output_dir = "documents/uae_cb_texts"
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch and save text content of each link
        for idx, link in enumerate(links, 1):
            try:
                print(f"Fetching content from: {link}")
                text = get_page_text(session, link)
                file_name = f"uae_cb_{idx}.txt"
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved content from {link} to {file_path}\n")
            except Exception as e:
                print(f"Failed to fetch {link}: {str(e)}")

if __name__ == "__main__":
    main()
