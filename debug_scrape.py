import requests
from bs4 import BeautifulSoup
import re

url = "https://www.thehindu.com/news/national/consensus-reached-to-revoke-suspension-of-eight-opposition-mps/article70750797.ece"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

try:
    print(f"Scraping {url}...")
    response = requests.get(url, headers=headers, timeout=15)
    print(f"Status: {response.status_code}")
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text().strip() for p in paragraphs])
    print(f"Extracted {len(text)} characters.")
    print(f"Preview: {text[:200]}...")
except Exception as e:
    print(f"Error: {e}")
