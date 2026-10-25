import numpy as np
import pyaudio
import librosa
import tensorflow as tf
import winsound
import time
import os
import sys

# ================= CẤU HÌNH =================
MODEL_PATH = "my_wakeword.tflite"
SAMPLE_RATE = 16000
DURATION = 1.5
SAMPLES = int(SAMPLE_RATE * DURATION)  # 24000 frame audio
CHUNK = 4000                           # Cập nhật thuật toán sau mỗi 0.25 giây
THRESHOLD = 0.85                       # Độ nhạy: > 85% là xác nhận (1.0 = 100%)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Không tìm thấy {MODEL_PATH}! Vui lòng chạy train_wakeword.py trước.")
    sys.exit(1)

    print("Dang nap mo hinh AI Wake-word...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(audio_data):
    """Trích xuất MFCC và đưa vào CNN suy luận thời gian thực"""
    # 1. Trích xuất MFCC y hệt như lúc Train
    mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=20, n_fft=512, hop_length=256)
    
    # [QUAN TRỌNG] Chuẩn hóa (Normalize) để khớp với lúc Train
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    
    # 2. Định dạng lại mảng theo đúng đầu vào của TFLite (1, 20, 94, 1)
    input_shape = input_details[0]['shape']
    target_time_steps = input_shape[2]
    
    if mfcc.shape[1] > target_time_steps:
        mfcc = mfcc[:, :target_time_steps]
    elif mfcc.shape[1] < target_time_steps:
        mfcc = np.pad(mfcc, ((0,0), (0, target_time_steps - mfcc.shape[1])))
        
    mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1).astype(np.float32)
    
    # 3. Phán đoán (Inference)
    interpreter.set_tensor(input_details[0]['index'], mfcc)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output[0][0] # Trả về % xác suất

def main():
    p = pyaudio.PyAudio()
    # Dùng paFloat32 để khớp hoàn toàn với định dạng chuẩn của thư viện librosa
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    print("="*60)
    print("HE THONG CUSTOM WAKE-WORD DA KICH HOAT!")
    print("="*60)
    print("Hay thu noi 'Bac si oi' hoac lam on de test.\n")

    # Tạo một dải băng chuyền âm thanh dài đúng 1.5 giây (24000 mẫu)
    buffer = np.zeros(SAMPLES, dtype=np.float32)
    last_detect_time = 0

    try:
        while True:
            # Lấy 0.25 giây âm thanh mới nhất
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Cuộn băng chuyền: Bỏ 0.25s cũ nhất ở bên trái, nhét 0.25s mới vào bên phải
            buffer = np.roll(buffer, -CHUNK)
            buffer[-CHUNK:] = audio_chunk
            
            # Tính toán âm lượng (VAD cơ bản)
            volume = np.sqrt(np.mean(buffer**2))
            
            # Nếu âm lượng đủ lớn (có tiếng động) thì mới đưa cho AI phân tích
            if volume > 0.005: 
                prob = predict(buffer)
                
                if prob > THRESHOLD:
                    now = time.time()
                    if now - last_detect_time > 2.0: # Cooldown 2 giây để không bíp liên tục
                        print(f"\r[PHAT HIEN TU KHOA] BIP! (Do tu tin: {prob*100:.1f}%)" + " "*10)
                        winsound.Beep(1000, 500)
                        last_detect_time = now
                        
                        # Xóa băng chuyền để tránh tàn dư âm thanh cũ
                        buffer = np.zeros(SAMPLES, dtype=np.float32)
                else:
                    # In ra xác suất để bạn dễ dàng căn chỉnh THRESHOLD nếu cần
                    print(f"\rDang nghe... (Ti le giong: {prob*100:.1f}%)", end='', flush=True)
            else:
                print("\rDang im lang... (Tiet kiem CPU)           ", end='', flush=True)
                
    except KeyboardInterrupt:
        print("\n\nĐã dừng hệ thống.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
