"""
tests/test_api.py — Automated Unit & Edge-Case Tests
"""

from fastapi.testclient import TestClient
from app.main import app
from app.extractor import calculate_shannon_entropy, detect_brand_spoofing, detect_leetspeak

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_entropy_and_spoof_helpers():
    assert calculate_shannon_entropy("aaaaaa") == 0.0
    assert calculate_shannon_entropy("x89a7b6c5d4e") > 3.2
    assert detect_brand_spoofing("login.paypal.com") == 0
    assert detect_brand_spoofing("paypal.com.secure-verify.net") == 1
    assert detect_brand_spoofing("paypal-account-update.com") == 1
    assert detect_leetspeak("paypa1-secure.com") == 1
    assert detect_leetspeak("github.com") == 0


def test_benign_root_and_deep_urls():
    benign_urls = [
        "https://www.google.com/",
        "https://github.com/torvalds/linux/commits/master",
        "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster",
        "https://docs.python.org/3/library/urllib.parse.html",
        "https://accounts.google.com/o/oauth2/v2/auth?client_id=123&response_type=code",
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
    ]
    for url in benign_urls:
        res = client.post("/predict", json={"url": url})
        assert res.status_code == 200
        data = res.json()
        assert data["is_phishing"] is False, f"False positive on benign URL: {url}"
        assert data["classification"] == "BENIGN"


def test_phishing_attack_vectors():
    phishing_urls = [
        "http://192.168.42.19/paypal-login/index.php",
        "https://paypa1-secure-login.com/account/signin",
        "https://apple.com.id-verification-portal.xyz/login",
        "http://microsoft-office365-security-update.tk/auth/signin",
        "http://www.amazon.com@auth-4918.xyz/login/verify",
    ]
    for url in phishing_urls:
        res = client.post("/predict", json={"url": url})
        assert res.status_code == 200
        data = res.json()
        assert data["is_phishing"] is True, f"False negative on phishing URL: {url}"
        assert data["classification"] == "MALICIOUS"
        assert len(data["risk_factors"]) >= 1