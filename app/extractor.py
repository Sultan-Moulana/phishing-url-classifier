import math
import re
from urllib.parse import urlparse
from datetime import datetime
import whois

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "secure", "account", 
    "banking", "confirm", "signin", "wp-admin", "auth", "wallet"
]

def calculate_shannon_entropy(text: str) -> float:
    """Calculates the Shannon entropy of a string to detect random/DGA strings."""
    if not text:
        return 0.0
    probabilities = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum(p * math.log(p, 2.0) for p in probabilities)

def get_domain_age_days(domain: str, live_lookup: bool = False) -> int:
    """
    Queries WHOIS for domain age in days. 
    Defaults to heuristic simulation during fast offline training/CI testing.
    """
    if not live_lookup:
        # Heuristic fallback for offline/fast inference: 
        # High-entropy or hyphenated domains mimic newly registered phishing domains
        return 15 if ("-" in domain or len(domain) > 22) else 1200
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date and isinstance(creation_date, datetime):
            return max(0, (datetime.now() - creation_date).days)
    except Exception:
        pass
    return 0  # Treat unresolvable WHOIS as newly registered/suspicious

def extract_url_features(url: str, live_whois: bool = False) -> list[float]:
    """Extracts an 11-dimensional numerical feature vector from a raw URL."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    domain = parsed.netloc.lower()
    
    url_length = len(url)
    domain_length = len(domain)
    dot_count = url.count(".")
    hyphen_count = domain.count("-")
    at_symbol = 1 if "@" in url else 0
    has_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain.split(":")[0]) else 0
    is_https = 1 if parsed.scheme == "https" else 0
    digit_ratio = sum(c.isdigit() for c in url) / max(len(url), 1)
    entropy = round(calculate_shannon_entropy(domain), 4)
    keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower())
    domain_age = get_domain_age_days(domain, live_lookup=live_whois)

    return [
        float(url_length),
        float(domain_length),
        float(dot_count),
        float(hyphen_count),
        float(at_symbol),
        float(has_ip),
        float(is_https),
        round(float(digit_ratio), 4),
        float(entropy),
        float(keyword_count),
        float(domain_age)
    ]