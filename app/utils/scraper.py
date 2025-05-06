import requests
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import urlparse
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_news_from_url(url):
    """
    Scrape news article content from a given URL
    
    Args:
        url (str): URL of the news article
        
    Returns:
        str: Extracted article text or None if extraction failed
    """
    try:
        # Add http:// if missing
        if not url.startswith('http'):
            url = 'https://' + url
        
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid URL format: {url}")
            return None
            
        # Send request with appropriate headers
        headers = {
            'User-Agent': config.USER_AGENT
        }
        
        response = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        # Try to find article content based on common patterns
        article_text = ""
        
        # Method 1: Look for article or main tags
        article_tag = soup.find('article') or soup.find('main')
        if article_tag:
            paragraphs = article_tag.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Method 2: If no article found, look for common content div classes
        if not article_text:
            content_divs = soup.find_all(['div', 'section'], class_=re.compile(r'(content|article|post|story|text)'))
            if content_divs:
                for div in content_divs:
                    paragraphs = div.find_all('p')
                    if paragraphs and len(paragraphs) > 3:  # Assume real articles have multiple paragraphs
                        article_text = ' '.join([p.get_text().strip() for p in paragraphs])
                        break
        
        # Method 3: If still no content, get all paragraphs
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
        # Clean up the text
        article_text = clean_text(article_text)
        
        if not article_text or len(article_text.split()) < 20:
            logger.warning(f"Could not extract meaningful content from {url}")
            return None
            
        logger.info(f"Successfully extracted {len(article_text.split())} words from {url}")
        return article_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return None

def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters that aren't relevant
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    
    return text

def extract_metadata(url):
    """
    Extract metadata from a news article URL
    
    Args:
        url (str): URL of the news article
        
    Returns:
        dict: Metadata including title, description, author, etc.
    """
    try:
        # Add http:// if missing
        if not url.startswith('http'):
            url = 'https://' + url
            
        # Send request
        headers = {
            'User-Agent': config.USER_AGENT
        }
        
        response = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract metadata
        metadata = {}
        
        # Title
        metadata['title'] = soup.title.string if soup.title else None
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content')
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content')
        
        # Author
        author = soup.find('meta', attrs={'name': 'author'})
        if author:
            metadata['author'] = author.get('content')
        
        # Publication date
        pub_date = soup.find('meta', attrs={'property': 'article:published_time'})
        if pub_date:
            metadata['published_date'] = pub_date.get('content')
        
        # Open Graph data
        metadata['og_title'] = soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else None
        metadata['og_description'] = soup.find('meta', property='og:description')['content'] if soup.find('meta', property='og:description') else None
        metadata['og_image'] = soup.find('meta', property='og:image')['content'] if soup.find('meta', property='og:image') else None
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {str(e)}")
        return {}