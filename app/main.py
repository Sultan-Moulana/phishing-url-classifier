import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if __package__:
    from .extractor import extract_url_features
else:
    from extractor import extract_url_features

app = FastAPI(
    title="PhishGuard-ML API",
    description="Real-time Machine Learning Phishing URL Detection Engine (v4 Feed-Trained)",
    version="1.1.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "app/model.joblib")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class URLPayload(BaseModel):
    url: str = Field(..., min_length=4, examples=["https://paypa1-secure-login.com/account/signin"])
    live_whois: bool = Field(default=False, description="Enable live WHOIS age query")

class Telemetry(BaseModel):
    url_length: float
    domain_entropy: float
    suspicious_keywords: float
    has_ip_literal: bool
    estimated_domain_age_days: float
    suspicious_tld: bool
    spoof_or_typosquat_detected: bool

class PredictionResponse(BaseModel):
    url: str
    classification: str
    is_phishing: bool
    risk_score: float
    telemetry: Telemetry

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def classify_url(payload: URLPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not initialized.")

    features = extract_url_features(payload.url, live_whois=payload.live_whois)
    probabilities = model.predict_proba([features])[0]
    phishing_prob = float(probabilities[1])
    is_phish = phishing_prob >= 0.5

    return PredictionResponse(
        url=payload.url,
        classification="MALICIOUS" if is_phish else "BENIGN",
        is_phishing=is_phish,
        risk_score=round(phishing_prob, 4),
        telemetry=Telemetry(
            url_length=features[0],
            domain_entropy=features[8],
            suspicious_keywords=features[9],
            has_ip_literal=bool(features[5]),
            estimated_domain_age_days=features[10],
            suspicious_tld=bool(features[13]),
            spoof_or_typosquat_detected=bool(features[15]),
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)