"""
app/main.py — FastAPI Inference Engine
"""

import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.extractor import extract_url_features, get_live_domain_age_days
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="PhishGuard-ML API",
    description="Real-time Machine Learning Phishing URL Detection Engine with Explainable Telemetry",
    version="2.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "app/model.joblib")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


class URLPayload(BaseModel):
    url: str = Field(
        ...,
        min_length=4,
        examples=["https://paypa1-secure-account-update.com/login/verify"],
    )
    live_whois: bool = Field(
        default=False,
        description="Perform a live WHOIS lookup to check if the domain was registered < 30 days ago",
    )


class Telemetry(BaseModel):
    url_length: int
    hostname_length: int
    path_length: int
    subdomain_depth: int
    hostname_entropy: float
    suspicious_keyword_count: int
    has_ip_literal: bool
    high_risk_tld: bool
    brand_spoofing_detected: bool
    leetspeak_detected: bool
    whois_domain_age_days: int | None = None


class PredictionResponse(BaseModel):
    url: str
    classification: str
    is_phishing: bool
    risk_score: float
    risk_factors: list[str]
    telemetry: Telemetry


def _build_risk_factors(features: list[float], domain_age: int | None) -> list[str]:
    reasons: list[str] = []
    if features[9] == 1.0:
        reasons.append("URL uses a raw IP address literal instead of a domain name")
    if features[16] == 1.0:
        reasons.append("Target brand name spoofed inside subdomain or hyphenated domain")
    if features[17] == 1.0:
        reasons.append("Leetspeak/typosquatting character substitution detected in hostname")
    if features[15] == 1.0:
        reasons.append("Hosted on a high-risk/abused Top-Level Domain (TLD)")
    if features[8] == 1.0:
        reasons.append("Contains '@' userinfo redirect obfuscation")
    if features[6] >= 2.0:
        reasons.append(f"Multiple hyphens ({int(features[6])}) in hostname (combo-squatting indicator)")
    if features[14] >= 2.0:
        reasons.append(f"Contains {int(features[14])} credential-harvesting keywords in URL")
    if features[12] > 3.75:
        reasons.append(f"High hostname Shannon entropy ({features[12]} bits) indicating random/DGA string")
    if domain_age is not None and domain_age < 30:
        reasons.append(f"Newly registered domain via WHOIS ({domain_age} days old)")
    return reasons

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PhishGuard-ML | Threat Engine</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white min-h-screen flex items-center justify-center font-sans">
        <div class="max-w-3xl w-full p-8 bg-gray-800 rounded-2xl shadow-2xl border border-gray-700">
            <div class="flex items-center space-x-3 mb-2">
                <div class="w-3 h-3 bg-red-500 rounded-full"></div>
                <div class="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div class="w-3 h-3 bg-green-500 rounded-full"></div>
            </div>
            <h1 class="text-4xl font-extrabold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">
                PhishGuard-ML
            </h1>
            <p class="text-gray-400 mb-8">Real-time Phishing URL Classification Engine</p>
            
            <div class="flex space-x-4 mb-6">
                <input type="text" id="urlInput" placeholder="Paste a suspicious URL (e.g., https://paypa1-secure.com)" 
                       class="w-full px-5 py-4 rounded-xl bg-gray-900 border border-gray-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all font-mono text-sm">
                <button onclick="scanUrl()" class="px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 rounded-xl font-bold shadow-lg transition-all active:scale-95">
                    Analyze
                </button>
            </div>
            
            <div id="result" class="hidden overflow-hidden rounded-xl border transition-all duration-300">
                <div class="p-6 bg-gray-900/50">
                    <div class="flex justify-between items-center mb-4">
                        <h2 id="verdict" class="text-3xl font-black tracking-wide"></h2>
                        <span id="score" class="px-4 py-2 rounded-lg text-sm font-bold bg-gray-950 border border-gray-700 shadow-inner"></span>
                    </div>
                    <div class="border-t border-gray-700/50 pt-4 mt-2">
                        <h3 class="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Risk Telemetry</h3>
                        <ul id="factors" class="list-disc list-inside text-gray-300 space-y-2 text-sm font-mono"></ul>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function scanUrl() {
                const url = document.getElementById('urlInput').value;
                if (!url) return;
                
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: url, live_whois: false})
                });
                
                const data = await res.json();
                const resultDiv = document.getElementById('result');
                const verdict = document.getElementById('verdict');
                const score = document.getElementById('score');
                const factors = document.getElementById('factors');
                
                resultDiv.classList.remove('hidden', 'border-red-500/50', 'border-emerald-500/50', 'shadow-[0_0_30px_-5px_rgba(239,68,68,0.3)]', 'shadow-[0_0_30px_-5px_rgba(16,185,129,0.3)]');
                factors.innerHTML = '';
                
                if (data.is_phishing) {
                    resultDiv.classList.add('border-red-500/50', 'shadow-[0_0_30px_-5px_rgba(239,68,68,0.3)]');
                    verdict.textContent = 'MALICIOUS';
                    verdict.className = 'text-3xl font-black tracking-wide text-red-500';
                } else {
                    resultDiv.classList.add('border-emerald-500/50', 'shadow-[0_0_30px_-5px_rgba(16,185,129,0.3)]');
                    verdict.textContent = 'BENIGN';
                    verdict.className = 'text-3xl font-black tracking-wide text-emerald-500';
                }
                
                score.textContent = `Threat Score: ${(data.risk_score * 100).toFixed(1)}%`;
                
                if (data.risk_factors.length > 0) {
                    data.risk_factors.forEach(f => {
                        const li = document.createElement('li');
                        li.textContent = f;
                        factors.appendChild(li);
                    });
                } else {
                    factors.innerHTML = '<li class="text-gray-500 italic">No elevated risk indicators detected in structural telemetry.</li>';
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def classify_url(payload: URLPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not initialized.")

    features = extract_url_features(payload.url)
    ml_prob = float(model.predict_proba([features])[0][1])

    domain_age = get_live_domain_age_days(payload.url) if payload.live_whois else None
    if domain_age is not None and domain_age < 30:
        ml_prob = min(0.9999, ml_prob + 0.15)

    is_phish = ml_prob >= 0.50
    risk_factors = _build_risk_factors(features, domain_age)

    return PredictionResponse(
        url=payload.url,
        classification="MALICIOUS" if is_phish else "BENIGN",
        is_phishing=is_phish,
        risk_score=round(ml_prob, 4),
        risk_factors=risk_factors,
        telemetry=Telemetry(
            url_length=int(features[0]),
            hostname_length=int(features[1]),
            path_length=int(features[2]),
            subdomain_depth=int(features[4]),
            hostname_entropy=features[12],
            suspicious_keyword_count=int(features[14]),
            has_ip_literal=bool(features[9]),
            high_risk_tld=bool(features[15]),
            brand_spoofing_detected=bool(features[16]),
            leetspeak_detected=bool(features[17]),
            whois_domain_age_days=domain_age,
        ),
    )