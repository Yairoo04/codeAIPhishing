import re
import math
from urllib.parse import urlparse
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Hàm trích xuất đặc trưng URL
def shannon_entropy(data):
    if not data:
        return 0
    prob = [freq / len(data) for freq in Counter(data).values()]
    return -sum(p * math.log2(p) for p in prob if p > 0)

def extract_features(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query

        if not domain:
            raise ValueError(f"Invalid URL: {url}")

        phishing_keywords = ['login', 'secure', 'account', 'password', 'signin', 'update', 'verify', 'bank', 'confirm']
        domain_parts = domain.split('.')
        domain_letters = re.sub(r'[^a-zA-Z]', '', domain)
        domain_numbers = re.sub(r'[^0-9]', '', domain)
        query_entropy = shannon_entropy(query) if query else 0

        features = {
            'url_length': len(url),
            'num_special_chars': len(re.findall(r'[^a-zA-Z0-9]', url)),
            'is_https': 1 if parsed_url.scheme == "https" else 0,
            'num_digits': len(re.findall(r'\d', url)),
            'domain_length': len(domain),
            'num_subdomains': max(len(domain_parts) - 2, 0),
            'num_dashes': url.count('-'),
            'path_length': len(path),
            'query_length': len(query),
            'has_ip': 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0,
            'has_at_symbol': 1 if '@' in url else 0,
            'redirect_count': url.count('http') - 1,
            'num_letters_in_domain': len(domain_letters),
            'num_numbers_in_domain': len(domain_numbers),
            'letter_to_number_ratio': len(domain_letters) / len(domain_numbers) if len(domain_numbers) > 0 else 0,
            'has_phishing_keywords': 1 if any(kw in url.lower() for kw in phishing_keywords) else 0,
            'num_query_params': len(query.split('&')) if query else 0,
            'query_string_complexity': query_entropy,
            'unicode_in_url': 1 if any(ord(c) > 127 for c in url) else 0
        }

        return features

    except Exception as e:
        logger.error(f"Lỗi khi trích xuất đặc trưng từ URL '{url}': {e}")
        return {
            'url_length': -1,
            'num_special_chars': -1,
            'is_https': -1,
            'num_digits': -1,
            'domain_length': -1,
            'num_subdomains': -1,
            'num_dashes': -1,
            'path_length': -1,
            'query_length': -1,
            'has_ip': -1,
            'has_at_symbol': -1,
            'redirect_count': -1,
            'num_letters_in_domain': -1,
            'num_numbers_in_domain': -1,
            'letter_to_number_ratio': -1,
            'has_phishing_keywords': -1,
            'num_query_params': -1,
            'query_string_complexity': -1,
            'unicode_in_url': -1
        }