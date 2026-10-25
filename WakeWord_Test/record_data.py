import pyaudio
import wave
import os
import time

# CẤU HÌNH THU ÂM
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 1.5  # 1.5 giây là quá đủ để nói chữ "Bác sĩ ơi"

DATASET_DIR = "dataset"
POS_DIR = os.path.join(DATASET_DIR, "positive")
NEG_DIR = os.path.join(DATASET_DIR, "negative")

# Tạo thư mục nếu chưa có
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

def record_sample(filename, duration=RECORD_SECONDS):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"\n🔴 [ĐANG THU ÂM] Hãy nói trong {duration} giây...")
    frames = []

    # Thu âm liên tục trong `duration` giây
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("✅ [XONG] Đã lưu tệp!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Ghi ra file WAV
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    print("="*50)
    print("🎙️ CÔNG CỤ THU ÂM WAKE WORD BẰNG GIỌNG THẬT 🎙️")
    print("="*50)
    
    while True:
        print("\nCHỌN CHẾ ĐỘ:")
        print("1. Thu âm từ khóa 'Bác sĩ ơi' (Positive)")
        print("2. Thu âm tiếng ồn/từ ngẫu nhiên (Negative)")
        print("0. Thoát")
        
        choice = input("👉 Nhập lựa chọn (0/1/2): ").strip()
        
        if choice == '0':
            break
            
        elif choice == '1':
            count = len(os.listdir(POS_DIR))
            print(f"\n📁 Đã có {count} mẫu từ khóa.")
            print("Nhấn Enter để bắt đầu thu âm (nhấn Ctrl+C để quay lại).")
            try:
                while True:
                    input("👉 [Bấm ENTER rồi ĐỌC NGAY: 'Bác sĩ ơi']...")
                    filename = os.path.join(POS_DIR, f"pos_{int(time.time())}.wav")
                    record_sample(filename)
            except KeyboardInterrupt:
                pass
                
        elif choice == '2':
            count = len(os.listdir(NEG_DIR))
            print(f"\n📁 Đã có {count} mẫu tiếng ồn/từ ngẫu nhiên.")
            print("Mẹo: Đọc các từ vớ vẩn, ho, hắt xì, hoặc giữ im lặng.")
            print("Nhấn Enter để bắt đầu thu âm (nhấn Ctrl+C để quay lại).")
            try:
                while True:
                    input("👉 [Bấm ENTER rồi tạo tiếng ồn/đọc từ khác]...")
                    filename = os.path.join(NEG_DIR, f"neg_{int(time.time())}.wav")
                    record_sample(filename)
            except KeyboardInterrupt:
                pass

if __name__ == "__main__":
    main()
