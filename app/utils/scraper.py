import requests
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import urlparse
import config

# Configure logging
logger = logging.getLogger(__name__)

def get_session():
    """Create a requests session with common headers"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': config.USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    })
    return session

def scrape_news_from_url(url):
    """Backwards compatibility wrapper"""
    res = scrape_all(url)
    return res.get('text') if res else None

def extract_metadata(url):
    """Backwards compatibility wrapper"""
    res = scrape_all(url)
    return res.get('metadata') if res else {}

def scrape_all(url):
    """
    Scrape text and metadata in a single request for efficiency and reliability
    """
    try:
        if not url.startswith('http'):
            url = 'https://' + url
            
        logger.info(f"Targeted scraping: {url}")
        session = get_session()
        
        response = session.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Extract Metadata
        metadata = {
            'title': soup.title.string.strip() if soup.title else None,
            'source': urlparse(url).netloc
        }
        
        # Open Graph & Meta
        meta_mappings = {
            'description': ['description', 'og:description'],
            'author': ['author', 'article:author'],
            'published_date': ['article:published_time', 'pub_date', 'date'],
            'image': ['og:image']
        }
        
        for key, tags in meta_mappings.items():
            for tag in tags:
                found = soup.find('meta', attrs={'name': tag}) or soup.find('meta', attrs={'property': tag})
                if found and found.get('content'):
                    metadata[key] = found.get('content')
                    break
        
        # 2. Extract Article Text
        # Clean soup
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "ad"]):
            junk.extract()
            
        article_text = ""
        
        # Strategy A: Article/Main tags
        container = soup.find('article') or soup.find('main') or soup.find(id=re.compile(r'article|content|story'))
        if container:
            paragraphs = container.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
            
        # Strategy B: Common class patterns
        if len(article_text.split()) < 50:
            content_divs = soup.find_all(['div', 'section'], class_=re.compile(r'(content|article|post|story|body|text)', re.I))
            for div in content_divs:
                p_count = len(div.find_all('p'))
                if p_count > 2:
                    text = ' '.join([p.get_text().strip() for p in div.find_all('p')])
                    if len(text.split()) > len(article_text.split()):
                        article_text = text
        
        # Strategy C: Greedy paragraph collection
        if len(article_text.split()) < 50:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])

        article_text = clean_text(article_text)
        
        if not article_text or len(article_text.split()) < 20:
            logger.warning(f"Thin content extracted (<20 words) from {url}")
            return {"text": article_text, "metadata": metadata}

        logger.info(f"Scrape successful: {len(article_text.split())} words extracted.")
        return {
            "text": article_text,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Scrape failed for {url}: {str(e)}")
        return None

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove obvious non-article text like "Cookies policy", "Click here", etc.
    junk_patterns = [r'read more', r'follow us', r'copyright', r'all rights reserved', r'click here']
    for p in junk_patterns:
        text = re.sub(p, '', text, flags=re.I)
    return text