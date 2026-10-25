import math
import re
from datetime import datetime
from urllib.parse import urlparse
import whois

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "subdomain_depth",
    "dot_count_hostname",
    "hyphen_count_hostname",
    "underscore_count",
    "at_symbol",
    "has_ip_literal",
    "is_https",
    "digit_ratio_hostname",
    "hostname_entropy",
    "path_entropy",
    "suspicious_keyword_count",
    "high_risk_tld",
    "brand_in_subdomain_or_hyphen",
    "leetspeak_or_homoglyph",
]

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "update", "secure",
    "account", "banking", "confirm", "wp-admin", "wp-content", "wallet",
    "billing", "unlock", "suspend", "restore", "credential", "webscr",
    "authenticate", "recovery", "validate", "password"
]

HIGH_RISK_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work",
    "click", "link", "buzz", "rest", "fit", "icu", "cam", "support",
    "info", "biz", "ru", "cn", "pw", "cc", "ws"
}

TARGETED_BRANDS = {
    "paypal", "apple", "icloud", "google", "gmail", "microsoft", "office365",
    "outlook", "onedrive", "amazon", "aws", "netflix", "chase", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "binance", "coinbase", "metamask",
    "steam", "steamcommunity", "discord", "instagram", "facebook", "whatsapp",
    "linkedin", "dropbox", "dhl", "fedex", "usps", "ups", "adobe", "docusign",
    "stripe", "revolut", "wise", "venmo"
}

LEETSPEAK_PATTERNS = [
    r"paypa[1lI]", r"app[1I]e", r"g[0o]{2,3}g[1l]e", r"micr[0o]s[0o]ft",
    r"amaz[0o]n", r"arnazon", r"netf[1l]ix", r"netfl1x", r"faceb[0o]{2}k",
    r"[1l]inkedin", r"linkedln", r"rnicrosoft", r"vvellsfargo", r"c[0o]inbase",
    r"b[1i]nance", r"stearncornmunity", r"disccrd", r"instagrarn"
]

# Known legitimate multi-part public suffixes so co.uk or com.au aren't miscounted as subdomains
SECOND_LEVEL_TLDS = {"co", "com", "org", "net", "edu", "gov", "ac", "mil"}


def calculate_shannon_entropy(text: str) -> float:
    """Calculates Shannon entropy (bits per character) of a string."""
    if not text:
        return 0.0
    length = len(text)
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _split_domain_parts(hostname: str) -> tuple[str, str, str]:
    """
    Splits hostname into (subdomain_str, registrable_domain_label, tld).
    Example: 'login.secure.paypal.com.phish.xyz' -> ('login.secure.paypal.com', 'phish', 'xyz')
    Example: 'portal.ox.ac.uk' -> ('portal', 'ox', 'ac.uk')
    """
    parts = [p for p in hostname.split(".") if p]
    if len(parts) < 2:
        return "", hostname, ""
    if len(parts) >= 3 and parts[-2] in SECOND_LEVEL_TLDS and len(parts[-1]) == 2:
        tld = f"{parts[-2]}.{parts[-1]}"
        reg_label = parts[-3]
        sub = ".".join(parts[:-3])
    else:
        tld = parts[-1]
        reg_label = parts[-2]
        sub = ".".join(parts[:-2])
    return sub, reg_label, tld


def detect_brand_spoofing(hostname: str) -> int:
    """
    Returns 1 if a major targeted brand appears in the subdomain or inside a
    hyphenated registrable domain (e.g., 'paypal-secure.com' or 'paypal.com.verify.net'),
    while returning 0 for the legitimate domain itself (e.g., 'login.paypal.com').
    """
    sub, reg_label, _ = _split_domain_parts(hostname)

    for brand in TARGETED_BRANDS:
        # Case 1: Brand is in the registrable domain, but NOT an exact match
        # e.g., reg_label = 'paypal-login' or 'securepaypal' -> Spoof!
        # e.g., reg_label = 'paypal' -> Legitimate owner!
        if brand in reg_label and reg_label != brand:
            return 1
        # Case 2: Brand is in the subdomain, and the registrable domain is NOT the brand
        # e.g., 'paypal.com.account-update.net' (sub='paypal.com', reg='account-update') -> Spoof!
        if brand in sub and reg_label != brand:
            return 1
    return 0


def detect_leetspeak(hostname: str) -> int:
    """Detects character/digit substitutions targeting major brands or mid-word digits."""
    for pat in LEETSPEAK_PATTERNS:
        if re.search(pat, hostname):
            return 1
    _, reg_label, _ = _split_domain_parts(hostname)
    # Digit sandwiched between letters inside the registrable domain label (e.g., paypa1, off1ce)
    if re.search(r"[a-z]+\d+[a-z]+", reg_label):
        return 1
    return 0


def get_live_domain_age_days(url: str) -> int | None:
    """Optional live WHOIS query for API enrichment."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        hostname = parsed.netloc.lower().split(":")[0]
        w = whois.whois(hostname)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date and isinstance(creation_date, datetime):
            return max(0, (datetime.now() - creation_date).days)
    except Exception:
        pass
    return None


def extract_url_features(url: str) -> list[float]:
    """Extracts 18 deterministic lexical and structural features from a URL."""
    clean_url = url.strip()
    parsed = urlparse(clean_url if "://" in clean_url else f"http://{clean_url}")

    # Strip userinfo if '@' is used for credential/redirect obfuscation
    raw_netloc = parsed.netloc.lower()
    hostname = raw_netloc.split("@")[-1].split(":")[0]
    if hostname.startswith("www."):
        hostname = hostname[4:]

    path = parsed.path or "/"
    query = parsed.query or ""

    sub, reg_label, tld = _split_domain_parts(hostname)

    # 1-4: Lengths
    url_length = len(clean_url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    # 5: Subdomain depth
    is_ip = bool(
        re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname)
        or hostname.isdigit()
        or re.match(r"^0x[0-9a-f]+$", hostname)
    )
    subdomain_depth = 0 if (is_ip or not sub) else len(sub.split("."))

    # 6-9: Character counts
    dot_count_hostname = hostname.count(".")
    hyphen_count_hostname = hostname.count("-")
    underscore_count = clean_url.count("_")
    at_symbol = 1 if "@" in clean_url else 0

    # 10-11: Protocol & IP literal
    has_ip_literal = 1 if is_ip else 0
    is_https = 1 if parsed.scheme.lower() == "https" else 0

    # 12: Digit ratio in hostname
    digit_ratio_hostname = sum(c.isdigit() for c in hostname) / max(len(hostname), 1)

    # 13-14: Entropy
    hostname_entropy = round(calculate_shannon_entropy(reg_label + sub), 4)
    path_entropy = round(calculate_shannon_entropy(path), 4)

    # 15: Suspicious keywords (counted in subdomain + path + query, or hyphenated domain)
    lower_target = f"{sub}/{path}?{query}".lower()
    if "-" in reg_label:
        lower_target += f" {reg_label}"
    suspicious_keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower_target)

    # 16: High-risk TLD
    high_risk_tld = 1 if tld in HIGH_RISK_TLDS else 0

    # 17-18: Brand spoofing & leetspeak
    brand_spoof = detect_brand_spoofing(hostname)
    leetspeak = detect_leetspeak(hostname)

    return [
        float(url_length),
        float(hostname_length),
        float(path_length),
        float(query_length),
        float(subdomain_depth),
        float(dot_count_hostname),
        float(hyphen_count_hostname),
        float(underscore_count),
        float(at_symbol),
        float(has_ip_literal),
        float(is_https),
        round(float(digit_ratio_hostname), 4),
        float(hostname_entropy),
        float(path_entropy),
        float(suspicious_keyword_count),
        float(high_risk_tld),
        float(brand_spoof),
        float(leetspeak),
    ]