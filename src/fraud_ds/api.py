from __future__ import annotations
import json
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from .features import add_features

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "metadata.json"
app = FastAPI(title="Fraud Risk Decision API", version="0.1.0")

class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")
    TransactionAmt: float
    TransactionDT: float
    ProductCD: str | None = None


def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Run training first; model.joblib and metadata.json are required.")
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists() and META_PATH.exists()}

@app.post("/predict")
def predict(tx: Transaction):
    try:
        model, meta = load_artifacts()
        row = add_features(pd.DataFrame([tx.model_dump()]))
        for col in meta["features"]:
            if col not in row:
                row[col] = None
        p = float(model.predict_proba(row[meta["features"]])[:,1][0])
        decline, review = float(meta["threshold"]), float(meta["review_threshold"])
        decision = "decline" if p >= decline else ("review" if p >= review else "approve")
        return {"fraud_probability": p, "decision": decision, "model_version": "0.1.0"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
