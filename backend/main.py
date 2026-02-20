from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="Russian L2 AES API")

# Allow the frontend (running on a different port) to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class EssayInput(BaseModel):
    text: str
    model_type: str = "deep_coral" # For future switching logic

class ScoreResponse(BaseModel):
    numeric_score: int       # 0-7 scale from your thesis
    actfl_label: str        # "Intermediate Mid", etc.
    confidence: float

class InsightsResponse(BaseModel):
    metrics: Dict[str, float]
    feedback: List[str]

# --- Helper Mappings (Based on Table 3.1 in your thesis [cite: 2803]) ---
ACTFL_MAPPING = {
    0: "Novice Mid",
    1: "Novice High",
    2: "Intermediate Low",
    3: "Intermediate Mid",
    4: "Intermediate High",
    5: "Advanced Low",
    6: "Advanced Mid",
    7: "Advanced High"
}

# --- Endpoints ---

@app.post("/score", response_model=ScoreResponse)
async def predict_score(essay: EssayInput):
    """
    Simulates the XLM-R + CORAL Deep Ordinal Model.
    In the future, your model.predict() logic goes here.
    """
    # HARDCODED MVP LOGIC
    # Simulating a model prediction
    fake_pred = 4  # Intermediate High
    
    return {
        "numeric_score": fake_pred,
        "actfl_label": ACTFL_MAPPING.get(fake_pred, "Unknown"),
        "confidence": 0.868 # Your QWK score from the thesis!
    }

@app.post("/insights", response_model=InsightsResponse)
async def analyze_text(essay: EssayInput):
    """
    Simulates the Feature-Based Model (Linear/CORN).
    Calculates the 24 engineered features[cite: 2971].
    """
    # HARDCODED MVP LOGIC
    return {
        "metrics": {
            "sentence_len_mean": 12.5,
            "lexical_density": 0.55,
            "ratio_b2": 0.15, # B2 vocabulary coverage
            "avg_tree_depth": 4.2 
        },
        "feedback": [
            "Good use of B2 vocabulary.",
            "Syntactic complexity is consistent with Intermediate High."
        ]
    }

# To run: uvicorn main:app --reload