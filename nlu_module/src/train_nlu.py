import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIGURATION
# =========================
PROCESSED_DATA_PATH = "data/processed/cleaned_mock_data.csv"
MODEL_DIR = "models"
MODEL_H5_PATH = os.path.join(MODEL_DIR, "nlu_model.h5")
MODEL_TFLITE_PATH = os.path.join(MODEL_DIR, "nlu_model.tflite")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "keras_tokenizer.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.json")

VOCAB_SIZE = 5000
MAX_LEN = 30
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 16
EPOCHS = 30

def build_and_train():
    # 1. TẠO THƯ MỤC NẾU CHƯA CÓ
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 2. LOAD DATA
    # Lưu ý: Các path sẽ chạy theo root directory của project là d:\NCKH_1.0\nlu_module
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"[LỖI] Không tìm thấy file: {PROCESSED_DATA_PATH}")
        print("Vui lòng chạy file `src/preprocess.py` để sinh dữ liệu trước!")
        return
        
    print(f"[*] Đang load dữ liệu từ {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Lọc các dòng lỗi (NaN)
    df = df.dropna(subset=['sentence', 'intent'])
    sentences = df['sentence'].astype(str).tolist()
    labels = df['intent'].astype(str).tolist()
    
    # 3. LABEL ENCODING
    print("[*] Ánh xạ Intent sang dạng số (Label Encoding)...")
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Lưu ra file label_encoder.json
    label_mapping = {str(index): label for index, label in enumerate(label_encoder.classes_)}
    with open(LABEL_ENCODER_PATH, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)
    print(f"  -> Đã lưu: {LABEL_ENCODER_PATH}")
    
    # 4. TOKENIZATION
    print("[*] Đang xây dựng từ điển (Tokenization)...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    
    # Lưu ra file keras_tokenizer.json
    tokenizer_json = tokenizer.to_json()
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)
    print(f"  -> Đã lưu: {TOKENIZER_PATH}")
    
    # 5. PADDING SEQUENCES
    print("[*] Cân bằng độ dài câu (Padding)...")
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    y = np.array(integer_labels)
    
    # 6. BUILD MODEL
    print("\n[*] Xây dựng kiến trúc mô hình BiLSTM...")
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(LSTM_UNITS)),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # 7. TRAIN MODEL
    print("\n[*] Bắt đầu huấn luyện...")
    model.fit(
        padded_sequences, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # Lấy 20% dữ liệu để validation
        verbose=1
    )
    
    # 8. EXPORT H5
    print(f"\n[*] Lưu mô hình gốc (.h5) -> {MODEL_H5_PATH}...")
    model.save(MODEL_H5_PATH)
    
    # 9. EXPORT TFLITE
    print("[*] Chuyển đổi mô hình sang định dạng TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Tối ưu hóa kích thước cho thiết bị Edge
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Cài đặt cờ để hỗ trợ các ops phức tạp của Tensor (TensorFlow Select ops) - quan trọng cho RNN/LSTM
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Mặc định
        tf.lite.OpsSet.SELECT_TF_OPS # Cần thiết cho BiLSTM
    ]
    
    tflite_model = converter.convert()
    
    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
        
    print(f"  -> Đã xuất file TFLite: {MODEL_TFLITE_PATH}")
    
    print("\n" + "="*50)
    print("🚀 QUÁ TRÌNH HUẤN LUYỆN VÀ CHUYỂN ĐỔI HOÀN TẤT!")
    print("Bàn giao 3 file sau cho đội ngũ Android (nằm trong thư mục models/):")
    print(" 1.", MODEL_TFLITE_PATH)
    print(" 2.", TOKENIZER_PATH)
    print(" 3.", LABEL_ENCODER_PATH)
    print("="*50)

if __name__ == "__main__":
    build_and_train()
