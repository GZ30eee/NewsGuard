import re
import ipaddress
import socket
from urllib.parse import urlparse

def is_safe_url(url):
    """
    Check if a URL is safe to scrape (No private IP, loopback, or invalid schemes)
    """
    try:
        if not url.startswith(('http://', 'https://')):
            return False
            
        parsed = urlparse(url)
        if not parsed.netloc:
            return False
            
        # Resolve hostname to IP
        hostname = parsed.hostname
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        
        # Block private, reserved, and loopback IPs (SSRF protection)
        if (ip_obj.is_private or 
            ip_obj.is_loopback or 
            ip_obj.is_reserved or 
            ip_obj.is_link_local or
            ip_obj.is_multicast):
            return False
            
        return True
    except Exception:
        return False

def sanitize_text(text, max_length=50000):
    """Basic text sanitization and length limiting"""
    if not text:
        return ""
    # Limit length to prevent DoS
    text = text[:max_length]
    # Remove null characters
    text = text.replace('\x00', '')
    return text.strip()
