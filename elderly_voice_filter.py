import os
import time
import json
import soundfile as sf
import torch
import numpy as np
import subprocess
import imageio_ffmpeg

# ==========================================
# CẤU HÌNH THƯ MỤC LƯU TRỮ TRÊN Ổ D
# ==========================================
CACHE_DIR = "D:\\HF_Cache"
OUTPUT_DIR = "D:\\Elderly_Voices"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "models")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

from datasets import load_dataset, Audio
from transformers import pipeline

# ==========================================
# CÔNG CỤ LÁCH LUẬT GIẢI MÃ FFmpeg
# ==========================================
def decode_audio_in_memory(audio_bytes):
    """
    Tự động giải mã âm thanh từ byte thô bằng imageio_ffmpeg,
    vượt qua hoàn toàn lỗi thiếu DLL của torchcodec.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, 
        "-i", "pipe:0",       # Đọc dữ liệu từ bộ nhớ (stdin)
        "-f", "s16le",        # Xuất ra định dạng PCM 16-bit
        "-ac", "1",           # Kênh Mono
        "-ar", "16000",       # Tần số 16kHz
        "pipe:1"              # Xuất vào bộ nhớ (stdout)
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, _ = process.communicate(input=audio_bytes)
    
    # Chuyển đổi dữ liệu byte thành mảng số thực float32
    audio_array = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_array, 16000


# ==========================================
# CẤU HÌNH LOGIC LỌC
# ==========================================
TIME_LIMIT_MINUTES = 45
MIN_AGE = 60
DATASET_NAME = "linhtran92/viet_bud500"

print("="*60)
print(f"🚀 KHỞI ĐỘNG CÔNG CỤ LỌC GIỌNG NGƯỜI CAO TUỔI (PHIÊN BẢN BYPASS)")
print(f"🕒 Giới hạn thời gian chạy: {TIME_LIMIT_MINUTES} phút")
print(f"💾 Thư mục lưu kết quả: {OUTPUT_DIR}")
print("="*60)

# 1. Tải Pipeline đoán tuổi
print("\n[*] Đang nạp mô hình AI đoán tuổi...")
age_predictor = pipeline("audio-classification", model="audeering/wav2vec2-large-robust-24-ft-age-gender")

# 2. Kết nối tập dữ liệu dạng Streaming
# ==========================================
# THAY THẾ CHUỖI 'hf_xxx' BÊN DƯỚI BẰNG TOKEN CỦA BẠN
# ==========================================
HF_TOKEN = "" 

print(f"\n[*] Đang kết nối Streaming với tập dữ liệu {DATASET_NAME}...")
dataset = load_dataset(DATASET_NAME, split="train", streaming=True, token=HF_TOKEN)

# LÁCH LUẬT: Vô hiệu hóa tính năng tự động giải mã của thư viện datasets
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. Vòng lặp lắng nghe & Lọc
start_time = time.time()
found_count = 0
scanned_count = 0

print("\n[*] BẮT ĐẦU QUÉT DỮ LIỆU...\n")

try:
    for sample in dataset:
        scanned_count += 1
        
        # Kiểm tra thời gian
        elapsed_time = time.time() - start_time
        if elapsed_time > (TIME_LIMIT_MINUTES * 60):
            print(f"\n⏰ Đã hết {TIME_LIMIT_MINUTES} phút! Hệ thống tự động ngắt kết nối.")
            break

        # TỰ GIẢI MÃ ÂM THANH TRONG BỘ NHỚ (Né torchcodec)
        audio_bytes = sample["audio"]["bytes"]
        audio_array, sampling_rate = decode_audio_in_memory(audio_bytes)
        
        transcription = sample["transcription"]
        
        # Dự đoán tuổi
        predictions = age_predictor(audio_array)
        
        estimated_age = 0
        for p in predictions:
            if 'age' in p['label'].lower():
                estimated_age = p['score'] * 100 
            elif p['label'] == 'LABEL_0': 
                 estimated_age = p['score'] * 100

        if estimated_age == 0 and len(predictions) > 0:
            estimated_age = predictions[0]['score'] * 100

        print(f"\rĐã quét: {scanned_count} file | Tìm thấy: {found_count} giọng già | Thời gian chạy: {elapsed_time/60:.1f} phút", end="")

        if estimated_age >= MIN_AGE:
            found_count += 1
            file_name = f"elderly_voice_{found_count}"
            wav_path = os.path.join(OUTPUT_DIR, file_name + ".wav")
            txt_path = os.path.join(OUTPUT_DIR, file_name + ".txt")
            
            sf.write(wav_path, audio_array, sampling_rate)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcription)

except KeyboardInterrupt:
    print("\n\n⛔ Đã bị hủy bởi người dùng (Ctrl+C).")
except Exception as e:
    print(f"\n\n❌ Lỗi không xác định: {e}")

print("\n" + "="*60)
print("📊 BÁO CÁO KẾT QUẢ TỔNG HỢP")
print(f"Tổng số file đã nghe thử : {scanned_count}")
print(f"Số giọng người > 60 tuổi  : {found_count}")
print(f"Thư mục lưu trữ         : {OUTPUT_DIR}")
print("="*60)
