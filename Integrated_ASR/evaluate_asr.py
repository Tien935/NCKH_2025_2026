import os
import sys
import time
import numpy as np
import subprocess
import imageio_ffmpeg
import jiwer
from datasets import load_dataset, Audio

# Khai báo đường dẫn để import được module Whisper của dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.asr_whisper import load_model

def decode_audio_in_memory(audio_bytes):
    """Bypass FFmpeg y hệt như công cụ lọc giọng"""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, _ = process.communicate(input=audio_bytes)
    audio_array = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_array

def main():
    print("="*60)
    print("📊 BẮT ĐẦU CHẤM ĐIỂM MÔ HÌNH WHISPER (BASELINE)")
    print("="*60)
    
    # 1. Load Whisper Model (Sử dụng hàm đã định nghĩa trong dự án)
    model = load_model()
    
    # 2. Load Dataset
    # ==========================================
    # THAY THẾ CHUỖI 'hf_xxx' BÊN DƯỚI BẰNG TOKEN CỦA BẠN
    # ==========================================
    HF_TOKEN = "" 
    
    print("\n[*] Đang kết nối Streaming với Bud500...")
    dataset = load_dataset("linhtran92/viet_bud500", split="train", streaming=True, token=HF_TOKEN)
    dataset = dataset.cast_column("audio", Audio(decode=False)) # Tắt tự động giải mã
    
    NUM_SAMPLES = 20
    scanned = 0
    
    ground_truths = []
    hypotheses = []
    total_audio_duration = 0
    total_inference_time = 0
    
    print(f"\n[*] Đang lấy {NUM_SAMPLES} mẫu đầu tiên để chấm thi...\n")
    
    for sample in dataset:
        if scanned >= NUM_SAMPLES:
            break
            
        audio_bytes = sample["audio"]["bytes"]
        reference_text = sample["transcription"]
        
        # Giải mã ra mảng số
        audio_array = decode_audio_in_memory(audio_bytes)
        audio_duration = len(audio_array) / 16000.0
        
        # Bắt đầu bấm giờ
        start_t = time.time()
        
        # Yêu cầu AI dịch
        result = model.transcribe(audio_array, language="vi")
        hypothesis_text = result["text"].strip()
        
        # Kết thúc bấm giờ
        end_t = time.time()
        
        inference_time = end_t - start_t
        total_audio_duration += audio_duration
        total_inference_time += inference_time
        
        # Làm sạch chuỗi trước khi chấm điểm (đưa về chữ thường)
        ground_truths.append(reference_text.lower())
        hypotheses.append(hypothesis_text.lower())
        
        scanned += 1
        print(f"[{scanned}/{NUM_SAMPLES}] Thời gian suy luận: {inference_time:.2f}s (Audio dài: {audio_duration:.2f}s)")
        print(f"   => ĐÁP ÁN: {reference_text}")
        print(f"   => AI DỊCH: {hypothesis_text}")
        print("-" * 50)
        
    print("\n" + "="*60)
    print("🏆 BẢNG ĐIỂM TỔNG KẾT (CHỤP LẠI MÀN HÌNH NÀY)")
    print("="*60)
    
    # Tính toán sai số
    wer_score = jiwer.wer(ground_truths, hypotheses)
    cer_score = jiwer.cer(ground_truths, hypotheses)
    
    rtf = total_inference_time / total_audio_duration
    
    print(f"- Tổng số câu đã giải     : {NUM_SAMPLES} câu")
    print(f"- Tổng độ dài âm thanh    : {total_audio_duration:.2f} giây")
    print(f"- Tổng thời gian dịch     : {total_inference_time:.2f} giây")
    print(f"- Tốc độ xử lý (RTF)      : {rtf:.2f} (càng nhỏ càng tốt, < 1 là AI dịch nhanh hơn cả tốc độ người nói)")
    print(f"- WER (Tỷ lệ lỗi từ)      : {wer_score * 100:.2f} % (Càng thấp càng tốt)")
    print(f"- CER (Tỷ lệ lỗi ký tự)   : {cer_score * 100:.2f} % (Càng thấp càng tốt)")
    print("="*60)

if __name__ == "__main__":
    main()
