import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoConfig, AutoModel, AutoTokenizer

app = FastAPI(title="Russian L2 AES API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Data Models ---
class EssayInput(BaseModel):
    text: str
    model_type: str = "deep_coral"

class ScoreResponse(BaseModel):
    numeric_score: int       
    actfl_label: str        
    confidence: float

class InsightsResponse(BaseModel):
    metrics: Dict[str, float]
    feedback: List[str]

ACTFL_MAPPING = {
    0: "Novice Mid", 1: "Novice High", 2: "Intermediate Low", 3: "Intermediate Mid",
    4: "Intermediate High", 5: "Advanced Low", 6: "Advanced Mid", 7: "Advanced High"
}

# --- 2. PyTorch Model Architecture (From your thesis/notebook) ---
class XLMRCoralOrdinal(nn.Module):
    def __init__(self, config_path, num_classes):
        super(XLMRCoralOrdinal, self).__init__()
        # Load blueprint locally to prevent downloading
        self.config = AutoConfig.from_pretrained(config_path)
        # Initialize empty base model
        self.roberta = AutoModel.from_config(self.config)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.arange(num_classes - 1, dtype=torch.float32))

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        logits = logits + self.coral_bias
        return logits

# --- 3. Global Model Loading ---
MODEL_DIR = "./mvp_model"
WEIGHTS_PATH = f"{MODEL_DIR}/quantized_coral_model.pt"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print("Initializing model architecture...")
model = XLMRCoralOrdinal(MODEL_DIR, num_classes=8)

print("Loading trained weights...")
state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))

# Bulletproof loading: Handles both standard and quantized state dicts
try:
    model.load_state_dict(state_dict)
    print("✅ Standard weights loaded.")
except RuntimeError:
    print("Standard load failed. Attempting quantized load...")
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(state_dict)
    print("✅ Quantized weights loaded.")

model.eval()

# --- 4. Endpoints ---
@app.post("/score", response_model=ScoreResponse)
async def predict_score(essay: EssayInput):
    """
    Runs the actual XLM-R + CORAL inference on the incoming text.
    """
    # 1. Tokenize text
    inputs = tokenizer(
        essay.text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    
    # 2. Forward pass through XLM-R
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # 3. CORAL Logic: Sigmoid and count thresholds > 0.5
        probas = torch.sigmoid(logits)
        predict_levels = torch.sum(probas > 0.5, dim=1)
        numeric_score = predict_levels.item()
    
    return {
        "numeric_score": numeric_score,
        "actfl_label": ACTFL_MAPPING.get(numeric_score, "Unknown"),
        "confidence": 0.868 # Static thesis QWK metric for MVP
    }

@app.post("/insights", response_model=InsightsResponse)
async def analyze_text(essay: EssayInput):
    # Still hardcoded for now until we build v0.2.0!
    return {
        "metrics": {"sentence_len_mean": 12.5, "lexical_density": 0.55, "ratio_b2": 0.15, "avg_tree_depth": 4.2},
        "feedback": ["Good use of B2 vocabulary.", "Syntactic complexity is consistent with Intermediate High."]
    }