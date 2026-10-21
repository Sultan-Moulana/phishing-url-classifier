import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from app.extractor import extract_url_features

# Curated training corpus (0 = Benign, 1 = Phishing)
TRAINING_CORPUS = [
    # Benign URLs
    ("https://www.google.com/search?q=devsecops", 0),
    ("https://github.com/torvalds/linux", 0),
    ("https://docs.python.org/3/library/urllib.parse.html", 0),
    ("https://stackoverflow.com/questions/11227809", 0),
    ("https://www.wikipedia.org/", 0),
    ("https://aws.amazon.com/security/", 0),
    ("https://news.ycombinator.com/news", 0),
    ("https://pypi.org/project/fastapi/", 0),
    ("https://www.microsoft.com/en-us/security", 0),
    ("https://www.cloudflare.com/learning/security/what-is-phishing/", 0),
    ("https://developer.mozilla.org/en-US/docs/Web/HTTP", 0),
    ("https://www.apple.com/macbook-air/", 0),
    ("https://ubuntu.com/server/docs", 0),
    ("https://hub.docker.com/_/python", 0),
    ("https://www.linkedin.com/in/security-engineer/", 0),

    # Malicious / Phishing URLs
    ("http://192.168.42.19/login-apple-id-verify/index.php", 1),
    ("http://secure-update-paypal-account-confirm.tk/signin", 1),
    ("http://x89a7b6c5d4e-auth-banking-portal.xyz/wp-admin/login.php", 1),
    ("http://www.amazon.com@secure-login-update-49182.info/verify", 1),
    ("http://login-micros0ft-office365-security-update.com/auth", 1),
    ("http://94.142.241.111/wallet-connect-verify-seed/confirm.html", 1),
    ("http://chase-online-banking-security-alert-2910.net/login", 1),
    ("http://netf1ix-billing-update-account-hold.org/signin/verify", 1),
    ("http://q9z8x7w6v5-dhl-parcel-tracking-confirm.top/update", 1),
    ("http://wellsfargo-secure-account-verification-portal.biz/auth", 1),
    ("http://185.220.101.45/wp-admin/secure-login-update.php", 1),
    ("http://steam-community-free-skins-auth-login.ru/confirm", 1),
    ("http://binance-wallet-kyc-verify-urgent-88219.xyz/login", 1),
    ("http://icloud-find-my-iphone-verify-location@45.33.32.156/signin", 1),
    ("http://bankofamerica-alert-confirm-identity-9912.club/secure", 1),
]

def train():
    print("[*] Extracting lexical & entropy features from corpus...")
    X = [extract_url_features(url, live_whois=False) for url, _ in TRAINING_CORPUS]
    y = [label for _, label in TRAINING_CORPUS]

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), np.array(y), test_size=0.2, random_state=42, stratify=y
    )

    print("[*] Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\n[*] Holdout Evaluation Report:")
    print(classification_report(y_test, preds, target_names=["Benign", "Phishing"]))

    model_path = "app/model.joblib"
    joblib.dump(clf, model_path)
    print(f"[+] Serialized model saved to {model_path}")

if __name__ == "__main__":
    train()