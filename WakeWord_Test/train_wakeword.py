import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

# CẤU HÌNH DỮ LIỆU
DATASET_DIR = "dataset"
POS_DIR = os.path.join(DATASET_DIR, "positive")
NEG_DIR = os.path.join(DATASET_DIR, "negative")

SAMPLE_RATE = 16000
DURATION = 1.5
SAMPLES = int(SAMPLE_RATE * DURATION) # 24000 frame audio

def extract_mfcc(file_path):
    """Trích xuất đặc trưng giọng nói bằng kỹ thuật MFCC (Tương tự Porcupine)"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Cắt hoặc đệm (Padding) để tất cả file đều dài chính xác 1.5 giây
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = np.pad(y, (0, max(0, SAMPLES - len(y))))
    
    # Tính MFCC: Chia âm thanh thành các khung nhỏ, tìm ra 20 dải tần số đặc trưng nhất
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20, n_fft=512, hop_length=256)
    
    # [QUAN TRỌNG] Chuẩn hóa (Normalize) dữ liệu để CNN có thể học được
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    
    return mfcc

def load_data():
    X = []
    y = []
    
    print("Dang doc va phan tich Du lieu Positive (Tu khoa)...")
    for f in glob.glob(os.path.join(POS_DIR, "*.wav")):
        X.append(extract_mfcc(f))
        y.append(1) # Gắn nhãn 1 (Đúng là Wake word)
        
    print("Dang doc va phan tich Du lieu Negative (Tieng on)...")
    for f in glob.glob(os.path.join(NEG_DIR, "*.wav")):
        X.append(extract_mfcc(f))
        y.append(0) # Gắn nhãn 0 (Rác)
        
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        raise ValueError("LỖI: Thư mục dataset trống! Vui lòng chạy record_data.py trước.")
        
    # Reshape thành dạng Ảnh 2D để đưa vào mạng CNN (batch_size, n_mfcc, time_steps, channels)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    return X, y

def build_cnn_model(input_shape):
    """Kiến trúc Mạng CNN mini - Cực nhẹ để chạy ngầm trên điện thoại"""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2), # Ngăn chặn Overfitting (Học vẹt)
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # Sigmoid nhả ra % (0 đến 100%)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("="*60)
    print("BAT DAU HUAN LUYEN AI WAKE-WORD")
    print("="*60)
    
    X, y = load_data()
    print(f"\nTong so mau am thanh: {len(y)}")
    print(f"Kich thuoc dau vao cua 1 mau (Anh pho MFCC): {X.shape[1]}x{X.shape[2]}")
    
    # Chia 80% để Train, 20% để Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_cnn_model(input_shape=(X.shape[1], X.shape[2], 1))
    
    print("\n🚀 Đang chạy Máy học (Training)...")
    # Tăng số vòng lặp lên 50 để AI học kỹ hơn (do dữ liệu của chúng ta hơi ít)
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
    
    # 1. Lưu bản Keras chuẩn
    model.save("my_wakeword.h5")
    print("\nDa luu file goc: my_wakeword.h5")
    
    # 2. Lượng tử hóa và nén sang chuẩn TensorFlow Lite (Chuyên trị Android/IoT)
    print("Dang nen mo hinh sang dinh dang .tflite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open("my_wakeword.tflite", "wb") as f:
        f.write(tflite_model)
        
    # Tính dung lượng
    size_kb = os.path.getsize("my_wakeword.tflite") / 1024
    print(f"THANH CONG! Da tao ra file 'my_wakeword.tflite'.")
    print(f"Dung luong sieu nhe: {size_kb:.2f} KB (Cuc ky hoan hao de cay vao Android)")

if __name__ == "__main__":
    main()
