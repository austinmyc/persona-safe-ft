"""
Agreeableness (and other Big Five) scores via Minej/bert-base-personality.
Used by rewrite_user.py to verify that rewritten user input has lower agreeableness.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "Minej/bert-base-personality"
LABEL_NAMES = ["Extroversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]
AGREEABLENESS_IDX = 2

_model = None
_tokenizer = None


def _get_model(model_id: str = MODEL_ID):
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForSequenceClassification.from_pretrained(model_id)
        _model.eval()
    return _model, _tokenizer


def get_agreeableness(text: str, model_id: str = MODEL_ID) -> float:
    """Return agreeableness score in [0, 1]. Higher = more agreeable."""
    if not (text or str(text).strip()):
        return 0.0
    model, tokenizer = _get_model(model_id)
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits).numpy()
    return float(probs[AGREEABLENESS_IDX])


def get_personality(text: str, model_id: str = MODEL_ID) -> dict:
    """Return dict of all five Big Five traits (0–1)."""
    if not (text or str(text).strip()):
        return {k: 0.0 for k in LABEL_NAMES}
    model, tokenizer = _get_model(model_id)
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    probs = torch.sigmoid(logits).numpy()
    return {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}
