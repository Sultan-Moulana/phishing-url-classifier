import math
import re
from urllib.parse import urlparse
from datetime import datetime
import whois

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "wp-admin", "auth", "wallet", "billing",
    "unlock", "suspend", "restore", "invoice", "credential", "kyc"
]

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".club", ".info", ".biz", ".ru", ".work", ".click"
}

TARGETED_BRANDS = [
    "paypal", "apple", "google", "microsoft", "amazon", "netflix",
    "chase", "wellsfargo", "bankofamerica", "icloud", "steam", "binance"
]

def calculate_shannon_entropy(text: str) -> float:
    """Calculates Shannon entropy to detect DGA/random strings."""
    if not text:
        return 0.0
    probabilities = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum(p * math.log(p, 2.0) for p in probabilities)

def has_leetspeak_or_typosquat(domain: str) -> int:
    """Detects digit substitutions inside words (e.g., paypa1, netfl1x, micros0ft) or homoglyphs."""
    # Check for digit sandwiched between letters (e.g. a1b, l1x, s0f)
    if re.search(r"[a-z]+\d+[a-z]+", domain):
        return 1
    # Check for brand names embedded in multi-hyphen or subdomain-hijacked domains
    for brand in TARGETED_BRANDS:
        if brand in domain and not domain.endswith(f"{brand}.com"):
            return 1
    return 0

def get_domain_age_days(domain: str, live_lookup: bool = False) -> int:
    """
    Queries WHOIS for domain age in days when live_lookup=True.
    When live_lookup=False (bulk training), uses structural domain risk heuristics.
    """
    if not live_lookup:
        is_suspicious_tld = any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)
        if is_suspicious_tld or "-" in domain or len(domain) > 24 or has_leetspeak_or_typosquat(domain):
            return 14
        return 1800
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date and isinstance(creation_date, datetime):
            return max(0, (datetime.now() - creation_date).days)
    except Exception:
        pass
    return 0

def extract_url_features(url: str, live_whois: bool = False) -> list[float]:
    """
    Extracts a 16-dimensional feature vector aligned with v4 adversarial & feed training:
      [0]  url_length
      [1]  domain_length
      [2]  dot_count
      [3]  hyphen_count (in domain)
      [4]  at_symbol
      [5]  has_ip
      [6]  is_https
      [7]  digit_ratio (in domain)
      [8]  domain_entropy
      [9]  keyword_count
      [10] domain_age
      [11] path_length
      [12] subdomain_count
      [13] suspicious_tld
      [14] query_param_count
      [15] brand_or_leetspeak_spoof
    """
    clean_url = url.strip()
    parsed = urlparse(clean_url if "://" in clean_url else f"http://{clean_url}")
    domain = parsed.netloc.lower().split(":")[0]
    path = parsed.path
    query = parsed.query

    url_length = len(clean_url)
    domain_length = len(domain)
    dot_count = clean_url.count(".")
    hyphen_count = domain.count("-")
    at_symbol = 1 if "@" in clean_url else 0

    # Matches dotted IPv4 or pure decimal/integer IP representations (like http://3627734734/)
    has_ip = 1 if (
        re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) or domain.isdigit()
    ) else 0

    is_https = 1 if parsed.scheme == "https" else 0
    digit_ratio = sum(c.isdigit() for c in domain) / max(len(domain), 1)
    entropy = round(calculate_shannon_entropy(domain), 4)
    keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in clean_url.lower())
    domain_age = get_domain_age_days(domain, live_lookup=live_whois)

    path_length = len(path)
    subdomain_count = max(0, domain.count(".") - 1) if not has_ip else 0
    suspicious_tld = 1 if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0
    query_param_count = query.count("&") + (1 if "=" in query else 0)
    spoof_indicator = has_leetspeak_or_typosquat(domain)

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
        float(domain_age),
        float(path_length),
        float(subdomain_count),
        float(suspicious_tld),
        float(query_param_count),
        float(spoof_indicator),
    ]