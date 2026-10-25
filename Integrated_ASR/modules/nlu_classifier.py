import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_H5_PATH = os.path.join(MODEL_DIR, "nlu_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "keras_tokenizer.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.json")

MAX_LEN = 30
CONFIDENCE_THRESHOLD = 0.70

_model = None
_tokenizer = None
_label_mapping = None

def load_nlu():
    global _model, _tokenizer, _label_mapping
    
    if _model is not None:
        return
        
    print("Đang load NLU Model (BiLSTM)...")
    
    # 1. Load model
    _model = load_model(MODEL_H5_PATH)
    
    # 2. Load tokenizer
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    _tokenizer = tokenizer_from_json(tokenizer_json)
    
    # 3. Load label encoder
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        _label_mapping = json.load(f)

def predict_intent(text: str):
    if _model is None:
        load_nlu()
        
    # Chuẩn bị dữ liệu
    sequences = _tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Dự đoán
    predictions = _model.predict(padded_seq, verbose=0)[0]
    
    # Lấy nhãn cao nhất
    max_index = np.argmax(predictions)
    confidence = float(predictions[max_index])
    
    # Label encoder lưu theo kiểu { "0": "request_emergency", ... }
    intent_name = _label_mapping.get(str(max_index), "unknown")
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "fallback", confidence
        
    return intent_name, confidence
