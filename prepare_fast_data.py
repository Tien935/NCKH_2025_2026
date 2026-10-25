import os
import csv
import numpy as np
import subprocess
import imageio_ffmpeg
import soundfile as sf
from datasets import load_dataset, Audio

# Thư mục lưu dữ liệu để train (Toàn bộ trên ổ D)
OUTPUT_DIR = "D:\\Bud500_Fast_Train"
WAVS_DIR = os.path.join(OUTPUT_DIR, "wavs")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(WAVS_DIR, exist_ok=True)

def decode_audio_in_memory(audio_bytes):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, _ = process.communicate(input=audio_bytes)
    return np.frombuffer(out, dtype=np.int16)

def main():
    print("="*60)
    print("📦 BẮT ĐẦU ĐÓNG GÓI DỮ LIỆU HUẤN LUYỆN (1000 MẪU)")
    print("="*60)
    
    # ==========================================
    # THAY THẾ CHUỖI 'hf_xxx' BÊN DƯỚI BẰNG TOKEN CỦA BẠN
    # ==========================================
    HF_TOKEN = ""
    
    dataset = load_dataset("linhtran92/viet_bud500", split="train", streaming=True, token=HF_TOKEN)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    NUM_SAMPLES = 1000
    scanned = 0
    
    # Mở file metadata để ghi
    with open(METADATA_PATH, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcription"])
        
        for sample in dataset:
            if scanned >= NUM_SAMPLES:
                break
                
            audio_bytes = sample["audio"]["bytes"]
            text = sample["transcription"]
            
            # Giải mã Audio
            audio_array = decode_audio_in_memory(audio_bytes)
            
            # Lưu file .wav
            file_name = f"sample_{scanned:04d}.wav"
            wav_path = os.path.join(WAVS_DIR, file_name)
            sf.write(wav_path, audio_array, 16000)
            
            # Ghi vào metadata
            writer.writerow([file_name, text])
            
            scanned += 1
            if scanned % 50 == 0:
                print(f"[*] Đã trích xuất và lưu: {scanned}/{NUM_SAMPLES} file...")
                
    print("="*60)
    print("✅ ĐÓNG GÓI HOÀN TẤT!")
    print(f"👉 Dữ liệu đã được lưu tại: {OUTPUT_DIR}")
    print("👉 Hãy Nén (Zip) toàn bộ thư mục này lại và đưa lên Google Drive!")
    print("="*60)

if __name__ == "__main__":
    main()
