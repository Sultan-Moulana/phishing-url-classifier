"""
train_model.py — Phishing URL Classifier  (v4)
------------------------------------------------
Data strategy:
  Phishing  → OpenPhish + mitchellkrogza/Phishing.Database  (live feeds)
  Benign    → Tranco top-1M list (zip download, take top 3k domains)
              + hardcoded Alexa/auth/OAuth domains as guaranteed floor
  Class balance target: no worse than 5:1 phishing:benign
"""

import io
import os
import sys
import zipfile
import urllib.request
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app.extractor import extract_url_features

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Hardcoded benign domains  (guaranteed floor — used even if Tranco fails)
# ─────────────────────────────────────────────────────────────────────────────
_BENIGN_DOMAINS_HARDCODED = [
    # Search / portals
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com", "baidu.com",
    "yandex.com", "ask.com", "ecosia.org", "brave.com", "qwant.com",
    # Social
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
    "reddit.com", "pinterest.com", "tiktok.com", "snapchat.com",
    "tumblr.com", "quora.com", "discord.com", "telegram.org",
    "whatsapp.com", "signal.org", "mastodon.social", "threads.net",
    # Video / media
    "youtube.com", "netflix.com", "twitch.tv", "vimeo.com",
    "hulu.com", "disneyplus.com", "primevideo.com", "crunchyroll.com",
    "spotify.com", "soundcloud.com", "pandora.com", "apple.com",
    "peacocktv.com", "paramountplus.com", "hbomax.com", "espn.com",
    # E-commerce
    "amazon.com", "ebay.com", "etsy.com", "walmart.com", "target.com",
    "bestbuy.com", "newegg.com", "aliexpress.com", "shopify.com",
    "costco.com", "homedepot.com", "lowes.com", "macys.com",
    "wayfair.com", "chewy.com", "overstock.com", "zappos.com",
    # Finance / banking
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citi.com",
    "paypal.com", "venmo.com", "stripe.com", "intuit.com",
    "fidelity.com", "schwab.com", "vanguard.com", "coinbase.com",
    "capitalone.com", "usbank.com", "tdbank.com", "pnc.com",
    "ally.com", "discover.com", "americanexpress.com", "mastercard.com",
    "visa.com", "transferwise.com", "wise.com", "revolut.com",
    # Tech
    "microsoft.com", "github.com", "stackoverflow.com", "gitlab.com",
    "cloudflare.com", "digitalocean.com", "heroku.com", "vercel.com",
    "npmjs.com", "pypi.org", "docker.com", "kubernetes.io",
    "mozilla.org", "opera.com", "adobe.com", "oracle.com",
    "salesforce.com", "servicenow.com", "workday.com", "sap.com",
    # Cloud
    "aws.amazon.com", "cloud.google.com", "azure.microsoft.com",
    # Developer
    "developer.mozilla.org", "docs.python.org", "docs.github.com",
    "readthedocs.io", "devdocs.io", "npmjs.com", "hub.docker.com",
    # News
    "nytimes.com", "bbc.com", "cnn.com", "theguardian.com",
    "reuters.com", "apnews.com", "bloomberg.com", "wsj.com",
    "washingtonpost.com", "usatoday.com", "forbes.com", "time.com",
    "techcrunch.com", "theverge.com", "wired.com", "ars technica.com",
    "arstechnica.com", "zdnet.com", "cnet.com", "engadget.com",
    # Gov / edu
    "usa.gov", "irs.gov", "ssa.gov", "cdc.gov", "fda.gov",
    "nasa.gov", "nist.gov", "mit.edu", "stanford.edu", "harvard.edu",
    "berkeley.edu", "ox.ac.uk", "cam.ac.uk", "yale.edu", "columbia.edu",
    # Reference
    "wikipedia.org", "britannica.com", "wolframalpha.com",
    "scholar.google.com", "pubmed.ncbi.nlm.nih.gov", "arxiv.org",
    # Security
    "owasp.org", "cve.mitre.org", "nvd.nist.gov", "shodan.io",
    "virustotal.com", "haveibeenpwned.com", "malwarebytes.com",
    "crowdstrike.com", "paloaltonetworks.com", "cisco.com",
    # SaaS / collaboration
    "office.com", "notion.so", "atlassian.com", "slack.com",
    "zoom.us", "webex.com", "dropbox.com", "box.com",
    "trello.com", "asana.com", "monday.com", "clickup.com",
    "figma.com", "canva.com", "miro.com", "airtable.com",
    # Auth / SSO  ← look "suspicious" but are totally benign
    "login.microsoftonline.com", "accounts.google.com",
    "signin.aws.amazon.com", "auth.atlassian.com",
    "id.apple.com", "account.live.com", "my.yahoo.com",
    "auth0.com", "okta.com", "onelogin.com", "ping.com",
    "identity.apple.com", "appleid.apple.com",
    "myaccount.google.com", "security.google.com",
    # E-mail
    "mail.google.com", "outlook.live.com", "mail.yahoo.com",
    "proton.me", "fastmail.com", "zoho.com", "tutanota.com",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Adversarial / edge-case phishing corpus
# ─────────────────────────────────────────────────────────────────────────────
_ADVERSARIAL_PHISHING: list[tuple[str, int]] = [
    # Classic HTTP
    ("http://192.168.42.19/login-apple-id-verify/index.php", 1),
    ("http://secure-update-paypal-account-confirm.tk/signin", 1),
    ("http://x89a7b6c5d4e-auth-banking-portal.xyz/wp-admin/login.php", 1),
    ("http://www.amazon.com@secure-login-update-49182.info/verify", 1),
    ("http://login-micros0ft-office365-security-update.com/auth", 1),
    ("http://94.142.241.111/wallet-connect-verify-seed/confirm.html", 1),
    ("http://chase-online-banking-security-alert-2910.net/login", 1),
    ("http://netf1ix-billing-update-account-hold.org/signin/verify", 1),
    ("http://185.220.101.45/wp-admin/secure-login-update.php", 1),
    ("http://steam-community-free-skins-auth-login.ru/confirm", 1),
    ("http://binance-wallet-kyc-verify-urgent-88219.xyz/login", 1),
    ("http://icloud-find-my-iphone-verify-location@45.33.32.156/signin", 1),
    ("http://bankofamerica-alert-confirm-identity-9912.club/secure", 1),
    # HTTPS phishing  ← breaks naive http heuristic
    ("https://paypa1-secure-login.com/account/signin", 1),
    ("https://accounts.google.com.phish-domain.net/auth", 1),
    ("https://secure-paypal-billing-update-required.com/signin", 1),
    ("https://apple-id-verify-account-locked.com/unlock", 1),
    ("https://microsoft-office365-security-login.com/auth", 1),
    ("https://netflix-billing-suspended-update.com/resume", 1),
    ("https://amazon-prime-renewal-confirm.com/payment", 1),
    ("https://chase-bank-secure-alert-verify.com/login", 1),
    ("https://icloud-account-disabled-verify.com/restore", 1),
    ("https://steam-community-trade-offer-confirm.com/accept", 1),
    # Subdomain hijacking
    ("http://paypal.com.account-verify-login.net/secure", 1),
    ("http://apple.com.id-verification-required.xyz/login", 1),
    ("http://google.com.secure-account-update.info/auth", 1),
    ("http://microsoft.com.login-verification.club/signin", 1),
    ("http://amazon.com.prime-account-update.biz/verify", 1),
    # Typosquatting
    ("http://arnazon.com/deals/login", 1),
    ("http://rn-icrosoftonline.com/common/oauth2", 1),
    ("http://gooogle.com/accounts/signin", 1),
    ("http://linkedln.com/login", 1),
    ("http://paypol.com/signin", 1),
    ("http://netfl1x.com/account/signin", 1),
    ("http://appl3.com/id/login", 1),
    ("http://redd1t.com/account/login", 1),
    # Compromised-looking paths
    ("https://legit-small-business.com/wp-content/plugins/paypal-login/", 1),
    ("https://local-news-site.org/images/invoice/apple-id-verify/index.php", 1),
    ("https://community-forum.net/uploads/microsoft-login/secure/auth.php", 1),
    # Numeric domains
    ("http://3627734734/paypal-login/secure/account.html", 1),
    ("http://secure-1o9gk2x.top/banking/verify/credentials.php", 1),
    # Urgent free TLDs
    ("http://urgent-account-suspension-verify-now.ml/login", 1),
    ("http://your-parcel-is-waiting-click-here.ga/track", 1),
    ("http://congratulations-you-won-claim-prize.cf/claim", 1),
]

_ADVERSARIAL_BENIGN: list[tuple[str, int]] = [
    # OAuth / login flows that look "suspicious" but are benign
    ("https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=abc123&response_type=code", 0),
    ("https://accounts.google.com/o/oauth2/auth?redirect_uri=https://app.example.com&scope=email", 0),
    ("https://signin.aws.amazon.com/signin?redirect_uri=console.aws.amazon.com&client_id=arn:aws", 0),
    ("https://auth.atlassian.com/authorize?audience=api.atlassian.com&client_id=xyz", 0),
    ("https://github.com/login/oauth/authorize?client_id=xyz&scope=repo&state=random123", 0),
    ("https://mail.google.com/mail/u/0/#inbox", 0),
    ("https://portal.azure.com/#view/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade", 0),
    ("https://console.cloud.google.com/iam-admin/iam?project=my-project", 0),
    ("https://id.apple.com/account/manage", 0),
    ("https://myaccount.google.com/security?hl=en", 0),
    ("https://account.live.com/password/reset?mkt=EN-US", 0),
]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Benign URL sources
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_DIR = "data"
_TRANCO_CACHE = os.path.join(_CACHE_DIR, "tranco_top1m.csv")


def _fetch(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as exc:
        print(f"    [!] {url[:70]}… → {exc}")
        return None


def _load_tranco(max_domains: int = 3000) -> list[str]:
    """
    Download Tranco top-1M list (zip) and return the top `max_domains` domains.
    Falls back to local cache if download fails.
    Returns plain domain strings (no scheme).
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)

    domains: list[str] = []

    # Try cache first (saves bandwidth on re-runs)
    if os.path.exists(_TRANCO_CACHE):
        print("    → Using cached Tranco list")
        with open(_TRANCO_CACHE, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= max_domains:
                    break
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    domains.append(parts[1].strip())
        if domains:
            return domains

    print("    → Downloading Tranco top-1M…")
    raw = _fetch("https://tranco-list.eu/top-1m.csv.zip", timeout=60)
    if raw is None:
        print("    [!] Tranco download failed — falling back to hardcoded list")
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_bytes = zf.read("top-1m.csv")
        csv_text = csv_bytes.decode("utf-8", errors="ignore")

        # Cache for next run
        with open(_TRANCO_CACHE, "w", encoding="utf-8") as fh:
            fh.write(csv_text)

        for i, line in enumerate(csv_text.splitlines()):
            if i >= max_domains:
                break
            parts = line.strip().split(",")
            if len(parts) >= 2:
                domains.append(parts[1].strip())

        print(f"    → {len(domains):,} domains from Tranco")
        return domains

    except Exception as exc:
        print(f"    [!] Tranco parse error: {exc}")
        return []


def _build_benign_corpus(max_benign: int = 3000) -> list[tuple[str, int]]:
    """
    Returns (url, 0) pairs up to max_benign.
    Priority: Tranco download → hardcoded list.
    """
    print("  [*] Loading benign URLs…")
    tranco = _load_tranco(max_domains=max_benign)

    # Combine Tranco + hardcoded, deduplicate
    all_domains: list[str] = []
    seen_d: set[str] = set()
    for d in (tranco + _BENIGN_DOMAINS_HARDCODED):
        if d and d not in seen_d:
            seen_d.add(d)
            all_domains.append(d)

    corpus = [(f"https://{d}/", 0) for d in all_domains[:max_benign]]
    print(f"    → {len(corpus):,} benign URLs")
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Phishing feed downloader
# ─────────────────────────────────────────────────────────────────────────────
_PHISHING_FEEDS = [
    {
        "name": "OpenPhish",
        "url": "https://openphish.com/feed.txt",
        "fmt": "plaintext",
    },
    {
        "name": "Phishing.Database",
        "url": (
            "https://raw.githubusercontent.com/mitchellkrogza/"
            "Phishing.Database/master/phishing-links-ACTIVE.txt"
        ),
        "fmt": "plaintext",
    },
    {
        "name": "PhishStats",
        "url": "https://phishstats.info/phish_score.csv",
        "fmt": "phishstats_csv",
    },
]


def _parse_phishing_feed(raw: str, fmt: str) -> list[str]:
    urls: list[str] = []
    if fmt == "plaintext":
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("http"):
                urls.append(line)
    elif fmt == "phishstats_csv":
        import csv as _csv
        for row in _csv.reader(io.StringIO(raw)):
            if not row or row[0].startswith("#"):
                continue
            if len(row) >= 3:
                u = row[2].strip().strip('"')
                if u.startswith("http"):
                    urls.append(u)
    return urls


def _download_phishing(max_per_feed: int = 3000) -> list[tuple[str, int]]:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    collected: list[tuple[str, int]] = []

    for feed in _PHISHING_FEEDS:
        name  = feed["name"]
        cache = os.path.join(_CACHE_DIR, f"{name.lower().replace('.','_')}.txt")

        print(f"  [*] Fetching {name}…")
        raw_bytes = _fetch(feed["url"])
        raw = raw_bytes.decode("utf-8", errors="ignore") if raw_bytes else None

        if raw is None and os.path.exists(cache):
            print("      → using cache")
            with open(cache, encoding="utf-8") as fh:
                raw = fh.read()

        if raw is None:
            print("      → skipped")
            continue

        with open(cache, "w", encoding="utf-8") as fh:
            fh.write(raw)

        urls  = _parse_phishing_feed(raw, feed["fmt"])[:max_per_feed]
        added = [(u, 1) for u in urls]
        collected.extend(added)
        print(f"      → {len(added):,} URLs")

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def _safe_extract(url: str) -> list[float] | None:
    try:
        feats = extract_url_features(url, live_whois=False)
        return list(feats) if feats is not None else None
    except Exception as exc:
        print(f"[!] Skipping ({exc}): {url[:70]}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main training routine
# ─────────────────────────────────────────────────────────────────────────────
_MAX_RATIO = 5          # max phishing : benign ratio before downsampling
_MIN_CORPUS = 200       # bail out if fewer than this many URLs survive extraction


def train() -> None:
    print("[*] Building corpus…")

    # ── Benign ───────────────────────────────────────────────────────────────
    benign_corpus = _build_benign_corpus(max_benign=3000)

    # ── Phishing ─────────────────────────────────────────────────────────────
    print("[*] Downloading phishing feeds…")
    phishing_corpus = _download_phishing(max_per_feed=3000)

    # ── Combine with adversarial corpus ──────────────────────────────────────
    corpus: list[tuple[str, int]] = (
        benign_corpus
        + _ADVERSARIAL_BENIGN
        + phishing_corpus
        + _ADVERSARIAL_PHISHING
    )

    # Deduplicate
    seen: dict[str, int] = {}
    for url, lbl in corpus:
        if url not in seen:
            seen[url] = lbl
    corpus = list(seen.items())

    dist = Counter(lbl for _, lbl in corpus)
    print(f"\n    Raw corpus: {len(corpus):,} URLs  "
          f"(benign={dist[0]:,}  phishing={dist[1]:,})")

    # ── Balance: cap phishing at MAX_RATIO × benign ───────────────────────────
    n_benign   = dist[0]
    n_phishing = dist[1]
    max_phish  = n_benign * _MAX_RATIO

    if n_phishing > max_phish:
        print(f"    Downsampling phishing {n_phishing:,} → {max_phish:,}  "
              f"(ratio capped at {_MAX_RATIO}:1)")
        rng = np.random.default_rng(42)
        benign_rows   = [(u, l) for u, l in corpus if l == 0]
        phishing_rows = [(u, l) for u, l in corpus if l == 1]
        idx = rng.choice(len(phishing_rows), size=max_phish, replace=False)
        corpus = benign_rows + [phishing_rows[i] for i in idx]

    dist = Counter(lbl for _, lbl in corpus)
    print(f"    Final corpus: {len(corpus):,} URLs  "
          f"(benign={dist[0]:,}  phishing={dist[1]:,})  "
          f"ratio {dist[1]/max(dist[0],1):.1f}:1")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("\n[*] Extracting features…")
    X_raw, y_raw = [], []
    for url, label in corpus:
        feats = _safe_extract(url)
        if feats is not None:
            X_raw.append(feats)
            y_raw.append(label)

    X = np.array(X_raw, dtype=float)
    y = np.array(y_raw, dtype=int)
    print(f"    Feature matrix: {X.shape}  "
          f"(benign={int((y==0).sum()):,}  phishing={int((y==1).sum()):,})")

    if len(set(y)) < 2 or len(y) < _MIN_CORPUS:
        sys.exit(f"[!] Corpus too small ({len(y)}) or only one class — aborting.")

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    print("\n[*] 5-fold stratified cross-validation…")
    clf_cv = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for metric in ("f1", "roc_auc", "precision", "recall"):
        scores = cross_val_score(clf_cv, X, y, cv=cv, scoring=metric, n_jobs=-1)
        label  = metric.upper().replace("_", "-")
        print(f"    CV {label:>10}: {scores.mean():.4f} ± {scores.std():.4f}  "
              f"[{' '.join(f'{s:.3f}' for s in scores)}]")

    # ── Holdout evaluation ────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[*] Training final model (train={len(y_tr):,}  test={len(y_te):,})…")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    cm    = confusion_matrix(y_te, preds)

    print("\n[*] Holdout Evaluation:")
    print(classification_report(y_te, preds, target_names=["Benign", "Phishing"]))
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"\n    Confusion matrix  (rows=actual, cols=predicted):")
    print(f"                  Benign  Phishing")
    print(f"    Benign        {cm[0,0]:>6}  {cm[0,1]:>8}")
    print(f"    Phishing      {cm[1,0]:>6}  {cm[1,1]:>8}")

    fn_rate = cm[1, 0] / max(cm[1].sum(), 1)
    fp_rate = cm[0, 1] / max(cm[0].sum(), 1)
    print(f"\n    False-negative rate (missed phishing): {fn_rate:.2%}")
    print(f"    False-positive rate (benign flagged) : {fp_rate:.2%}")

    # ── Feature importances ───────────────────────────────────────────────────
    top = np.argsort(clf.feature_importances_)[::-1][:10]
    print("\n[*] Top-10 feature importances:")
    for rank, i in enumerate(top, 1):
        print(f"    {rank:>2}. feature[{i:>2}] = {clf.feature_importances_[i]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("app", exist_ok=True)
    model_path = os.path.join("app", "model.joblib")
    joblib.dump(clf, model_path)
    print(f"\n[+] Model saved → {model_path}")

    # ── Health check ──────────────────────────────────────────────────────────
    if fp_rate > 0.10:
        print(
            "\n[!] WARNING: False-positive rate > 10 % — the model is too aggressive."
            "\n    Add more benign URLs or lower max_depth."
        )
    if fn_rate > 0.05:
        print(
            "\n[!] WARNING: False-negative rate > 5 % — the model misses too much phishing."
            "\n    Check your phishing feed coverage."
        )


if __name__ == "__main__":
    train()