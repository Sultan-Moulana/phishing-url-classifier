import os
import random
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app.extractor import FEATURE_NAMES, extract_url_features

SEED = 42

LEGIT_DOMAINS = [
    "google.com", "github.com", "microsoft.com", "apple.com", "amazon.com",
    "wikipedia.org", "stackoverflow.com", "linkedin.com", "reddit.com", "youtube.com",
    "netflix.com", "nytimes.com", "bbc.co.uk", "theguardian.com", "reuters.com",
    "cloudflare.com", "digitalocean.com", "docker.com", "kubernetes.io", "python.org",
    "pypi.org", "npmjs.com", "mozilla.org", "ubuntu.com", "debian.org",
    "mit.edu", "stanford.edu", "harvard.edu", "ox.ac.uk", "cam.ac.uk",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "paypal.com", "stripe.com",
    "coinbase.com", "wise.com", "revolut.com", "fidelity.com", "vanguard.com",
    "atlassian.com", "slack.com", "notion.so", "figma.com", "canva.com",
    "dropbox.com", "zoom.us", "salesforce.com", "oracle.com", "adobe.com",
    "spotify.com", "twitch.tv", "discord.com", "steampowered.com", "epicgames.com",
    "walmart.com", "target.com", "bestbuy.com", "ebay.com", "etsy.com",
    "shopify.com", "ikea.com", "booking.com", "airbnb.com", "expedia.com",
    "nasa.gov", "nist.gov", "cdc.gov", "who.int", "arxiv.org",
]

LEGIT_SUBDOMAINS = [
    "docs", "developer", "api", "support", "help", "blog", "news",
    "status", "community", "learn", "portal", "console", "cloud", "mail"
]

LEGIT_PATHS = [
    "/",
    "/about",
    "/pricing",
    "/contact-us",
    "/docs/getting-started/installation",
    "/docs/api/v2/reference/authentication",
    "/blog/2026/10/securing-modern-container-pipelines",
    "/articles/machine-learning-best-practices-production",
    "/products/electronics/wireless-noise-canceling-headphones-xm5",
    "/questions/11227809/why-is-processing-a-sorted-array-faster",
    "/repos/torvalds/linux/commits/master",
    "/search?q=devsecops+ci+cd+pipeline&page=2&sort=relevance",
    "/watch?v=dQw4w9WgXcQ&ab_channel=OfficialArtist",
    "/user/profile/settings/notifications",
    "/courses/cybersecurity-fundamentals/modules/network-defense",
    "/en-us/windows/security/threat-protection/overview",
    "/release-notes/v4.12.0/changelog.html",
]

LEGIT_AUTH_URLS = [
    "https://accounts.google.com/ServiceLogin?hl=en&passive=true&continue=https://www.google.com/",
    "https://accounts.google.com/o/oauth2/v2/auth?client_id=48192.apps.googleusercontent.com&response_type=code&scope=openid%20email",
    "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=9a8b7c&response_type=code&redirect_uri=https://portal.azure.com",
    "https://github.com/login/oauth/authorize?client_id=Iv1.8a7b6c5d4e&scope=read:user",
    "https://github.com/login?return_to=%2Ftorvalds%2Flinux",
    "https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fconsole.aws.amazon.com",
    "https://id.apple.com/auth/authorize?client_id=com.apple.account",
    "https://auth.atlassian.com/authorize?audience=api.atlassian.com&prompt=login",
    "https://www.paypal.com/signin?returnUri=https%3A%2F%2Fwww.paypal.com%2Fmyaccount",
    "https://www.linkedin.com/uas/login?session_redirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2F",
    "https://myaccount.google.com/security-checkup",
    "https://store.steampowered.com/login/?redir=app%2F730",
    "https://discord.com/login?redirect_to=%2Fchannels%2F%40me",
    "https://www.dropbox.com/login?cont=https%3A%2F%2Fwww.dropbox.com%2Fhome",
]

PHISH_BRANDS = [
    "paypal", "apple", "icloud", "google", "microsoft", "office365",
    "amazon", "netflix", "chase", "wellsfargo", "bankofamerica",
    "binance", "coinbase", "metamask", "steam", "steamcommunity",
    "dhl", "fedex", "usps", "docusign", "linkedin", "instagram",
    "facebook", "whatsapp", "dropbox", "adobe", "outlook", "revolut"
]

PHISH_WORDS = [
    "secure", "login", "signin", "verify", "verification", "update",
    "account", "billing", "confirm", "unlock", "alert", "support",
    "auth", "portal", "recovery", "suspend", "validation", "security"
]

PHISH_TLDS = [
    "com", "net", "org", "info", "biz", "xyz", "top", "club",
    "tk", "ml", "ga", "cf", "gq", "work", "click", "link", "ru", "icu", "buzz"
]

LEET_DOMAINS = [
    "paypa1.com", "paypa1-secure.com", "g00gle-accounts.com", "micr0s0ft-login.com",
    "arnazon-prime.com", "netf1ix-billing.com", "netfl1x.com", "app1e-id-support.com",
    "faceb00k-security.com", "linkedln-auth.com", "rnicrosoftonline.com",
    "c0inbase-wallet.com", "b1nance-verify.com", "stearncornmunity.com",
    "vvellsfargo-online.com", "instagrarn-help.com", "disccrd-nitro.com"
]

COMPROMISED_DOMAINS = [
    "green-valley-dentistry.com", "bella-italia-bistro.org", "midwest-plumbing-supply.net",
    "sunnyvale-youth-soccer.org", "heritage-woodworking-shop.com", "metro-auto-repair.net",
    "old-town-bakery-cafe.com", "pacific-coast-landscaping.org", "alpine-ski-rentals.net"
]


def generate_dataset(rng: random.Random) -> list[tuple[str, int]]:
    dataset: list[tuple[str, int]] = []

    # -------------------------------------------------------------------------
    # 1. Generate 2,200 Realistic Benign URLs (Root, Deep Paths, Subdomains, Auth)
    # -------------------------------------------------------------------------
    for domain in LEGIT_DOMAINS:
        # Bare root domain
        dataset.append((f"https://{domain}/", 0))
        dataset.append((f"https://www.{domain}/", 0))

        # Deep paths on main domain
        for path in rng.sample(LEGIT_PATHS, k=10):
            dataset.append((f"https://{domain}{path}", 0))

        # Deep paths on legitimate subdomains
        for sub in rng.sample(LEGIT_SUBDOMAINS, k=5):
            path = rng.choice(LEGIT_PATHS)
            dataset.append((f"https://{sub}.{domain}{path}", 0))

    # Add realistic mid-size company / blog / SaaS benign domains (with single hyphens)
    benign_SaaS = [
        "scikit-learn.org", "paloaltonetworks.com", "haveibeenpwned.com",
        "ars-technica.com", "Css-tricks.com", "smashingmagazine.com",
        "hacker-news.firebaseio.com", "react-hook-form.com", "tailwind-ui.com",
        "start-bootstrap.com", "material-ui.com", "spring-boot.io"
    ]
    for domain in benign_SaaS:
        dataset.append((f"https://{domain.lower()}/", 0))
        for path in rng.sample(LEGIT_PATHS, k=8):
            dataset.append((f"https://{domain.lower()}{path}", 0))

    # Multiply legitimate OAuth/Login URLs so the model learns legit logins != phishing
    for auth_url in LEGIT_AUTH_URLS * 18:
        dataset.append((auth_url, 0))

    # -------------------------------------------------------------------------
    # 2. Generate 2,200 Diverse Phishing URLs across 7 Real-World Attack Vectors
    # -------------------------------------------------------------------------

    # Vector A: Combo-squatting / Hyphenated Brand Domains (HTTPS & HTTP)
    for _ in range(550):
        brand = rng.choice(PHISH_BRANDS)
        w1, w2 = rng.sample(PHISH_WORDS, 2)
        tld = rng.choice(PHISH_TLDS)
        scheme = rng.choice(["http", "https"])
        pattern = rng.choice([
            f"{scheme}://{brand}-{w1}-{w2}.{tld}/{w1}/index.php",
            f"{scheme}://{w1}-{brand}-{w2}.{tld}/auth/login",
            f"{scheme}://{brand}-{w1}.{tld}/account/{w2}?session={rng.randint(10000, 99999)}",
            f"{scheme}://{w1}-{w2}-{brand}-{rng.randint(10, 999)}.{tld}/signin",
        ])
        dataset.append((pattern, 1))

    # Vector B: Subdomain Hijacking / Brand in Subdomain
    for _ in range(450):
        brand = rng.choice(PHISH_BRANDS)
        w1, w2 = rng.sample(PHISH_WORDS, 2)
        tld = rng.choice(PHISH_TLDS)
        scheme = rng.choice(["http", "https"])
        host_base = rng.choice(["account-service", "auth-portal", "user-check", "session-id", "cloud-gate"])
        pattern = rng.choice([
            f"{scheme}://{brand}.com.{w1}-{w2}.{tld}/login",
            f"{scheme}://login.{brand}.com.{host_base}.{tld}/{w1}/auth.php",
            f"{scheme}://{brand}.{w1}.{host_base}.{tld}/verify?token={rng.randint(100000, 999999)}",
        ])
        dataset.append((pattern, 1))

    # Vector C: Leetspeak & Typosquatting (Short domains, HTTPS & HTTP)
    for _ in range(300):
        leet_host = rng.choice(LEET_DOMAINS)
        w1 = rng.choice(PHISH_WORDS)
        scheme = rng.choice(["http", "https"])
        path = rng.choice([
            f"/{w1}/login",
            "/account/signin",
            "/common/oauth2/authorize",
            "/myaccount/security",
            "/"
        ])
        dataset.append((f"{scheme}://{leet_host}{path}", 1))

    # Vector D: Raw IPv4 Literal Phishing
    for _ in range(250):
        ip = f"{rng.randint(11, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        brand = rng.choice(PHISH_BRANDS)
        w1 = rng.choice(PHISH_WORDS)
        port = rng.choice(["", ":8080", ":8443"])
        dataset.append((f"http://{ip}{port}/{brand}-{w1}/login.php", 1))

    # Vector E: Compromised Legitimate WordPress/cPanel Sites Hosting Phishing Kits
    for _ in range(250):
        comp_host = rng.choice(COMPROMISED_DOMAINS)
        brand = rng.choice(PHISH_BRANDS)
        w1 = rng.choice(PHISH_WORDS)
        scheme = rng.choice(["http", "https"])
        path = rng.choice([
            f"/wp-content/plugins/{brand}-{w1}/index.php",
            f"/wp-admin/includes/{brand}/login/verify.html",
            f"/wp-includes/SimplePie/{brand}-account-update/signin.php",
            f"/.well-known/acme-challenge/{brand}-login/auth.php",
        ])
        dataset.append((f"{scheme}://{comp_host}{path}", 1))

    # Vector F: High-Entropy / Random DGA Hosts on Abused Free TLDs
    for _ in range(250):
        rand_str = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=rng.randint(10, 18)))
        tld = rng.choice(["tk", "ml", "ga", "cf", "gq", "xyz", "top", "icu", "buzz", "rest"])
        brand = rng.choice(PHISH_BRANDS)
        w1 = rng.choice(PHISH_WORDS)
        scheme = rng.choice(["http", "https"])
        dataset.append((f"{scheme}://{rand_str}.{tld}/{brand}/{w1}.html", 1))

    # Vector G: '@' Symbol Userinfo Redirect Obfuscation
    for _ in range(150):
        brand = rng.choice(PHISH_BRANDS)
        tld = rng.choice(["xyz", "top", "info", "ru", "club"])
        rand_host = f"auth-{rng.randint(1000, 9999)}.{tld}"
        dataset.append((f"http://www.{brand}.com@{rand_host}/login/verify", 1))

    return dataset


def train() -> None:
    rng = random.Random(SEED)
    print("[*] Generating balanced structural URL corpus (v5)...")
    raw_corpus = generate_dataset(rng)

    # Deduplicate while preserving order
    seen: dict[str, int] = {}
    for url, label in raw_corpus:
        seen[url] = label
    corpus = list(seen.items())
    rng.shuffle(corpus)

    X = np.array([extract_url_features(u) for u, _ in corpus], dtype=float)
    y = np.array([lbl for _, lbl in corpus], dtype=int)

    n_benign = int((y == 0).sum())
    n_phish = int((y == 1).sum())
    print(f"    Total unique URLs : {len(y):,} (Benign: {n_benign:,} | Phishing: {n_phish:,})")
    print(f"    Feature matrix    : {X.shape}")

    # Build a soft-voting ensemble of Random Forest + Gradient Boosting
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.08,
        max_depth=5,
        random_state=SEED,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )

    print("\n[*] Running 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for metric in ("f1", "roc_auc", "precision", "recall"):
        scores = cross_val_score(ensemble, X, y, cv=cv, scoring=metric, n_jobs=-1)
        print(f"    CV {metric.upper():>9}: {scores.mean():.4f} ± {scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    print(f"\n[*] Training Ensemble Classifier (train={len(y_train):,}, test={len(y_test):,})...")
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    probs = ensemble.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("\n[*] Holdout Evaluation Report:")
    print(classification_report(y_test, preds, target_names=["Benign", "Phishing"], digits=4))
    print(f"    ROC-AUC Score : {auc:.4f}")
    print(f"    Confusion Matrix: TN={cm[0,0]} | FP={cm[0,1]} | FN={cm[1,0]} | TP={cm[1,1]}")

    # Inspect RF feature importances inside the fitted ensemble
    fitted_rf = ensemble.named_estimators_["rf"]
    top_indices = np.argsort(fitted_rf.feature_importances_)[::-1][:10]
    print("\n[*] Top 10 Most Important Features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:>2}. {FEATURE_NAMES[idx]:<28} = {fitted_rf.feature_importances_[idx]:.4f}")

    os.makedirs("app", exist_ok=True)
    model_path = os.path.join("app", "model.joblib")
    joblib.dump(ensemble, model_path)
    print(f"\n[+] Saved production model artifact -> {model_path}")


if __name__ == "__main__":
    train()