import os
import time
import jiwer
import numpy as np
import random
from gtts import gTTS
import subprocess
from modules.asr_whisper import load_model, transcribe

# Khởi tạo một số câu y tế ĐỂ KIỂM TRA (Có một vài câu hơi khác để xem AI có khả năng khái quát hóa không)
TEST_SENTENCES = [
    "trợ lý ơi tôi đau đầu quá",
    "cháu ơi hãy gọi cấp cứu cho tôi",
    "alo tôi thấy chóng mặt và buồn nôn",
    "huyết áp của tôi tự nhiên tăng cao",
    "con ơi lấy cho tôi vỉ thuốc trợ tim",
    "bác sĩ ơi tôi bị khó thở quá",
    "bạn ơi ngực tôi đau thắt lại",
    "tôi bị ngã đau quá",
    "nghe này cứu tôi với",
    "trợ lý ơi đo nhịp tim giúp tôi",
    "nhắc tôi uống thuốc ngay lập tức",
    "cô ơi hướng dẫn tôi cách sơ cứu",
    "tôi cần đi bệnh viện khẩn cấp",
    "đầu tôi đau như búa bổ",
    "tôi bị ho và sốt từ chiều",
    "tôi buồn nôn quá",
    "tôi bị đau bụng dữ dội",
    "tim tôi đập nhanh quá",
    "mang cho tôi ly nước ấm",
    "tôi cần uống thuốc giảm đau ngay"
]

TEST_DIR = os.path.join(os.path.dirname(__file__), "data", "test_medical")
os.makedirs(TEST_DIR, exist_ok=True)

def generate_test_audio():
    print("="*60)
    print("🛠️ ĐANG TẠO 20 BÀI THI Y TẾ MỚI ĐỂ KIỂM TRA AI...")
    
    test_data = []
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    for i, text in enumerate(TEST_SENTENCES):
        mp3_path = os.path.join(TEST_DIR, f"test_{i}.mp3")
        wav_path = os.path.join(TEST_DIR, f"test_{i}.wav")
        
        # 1. Tạo giọng đọc bằng Google TTS
        tts = gTTS(text, lang='vi', slow=False)
        tts.save(mp3_path)
        
        # 2. Làm chậm giọng đọc (tương tự lúc học)
        tempo = round(random.uniform(0.75, 0.9), 2)
        cmd = [
            ffmpeg_exe, "-y", "-i", mp3_path,
            "-filter:a", f"atempo={tempo}",
            "-ac", "1", "-ar", "16000", wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(mp3_path)
        
        # Lấy độ dài audio
        import soundfile as sf
        info = sf.info(wav_path)
        duration = info.duration
        
        test_data.append({
            "path": wav_path,
            "reference": text,
            "duration": duration
        })
        print(f"  Đã tạo đề thi {i+1}/20...")
        
    return test_data

def evaluate():
    print("="*60)
    print("🚀 BẮT ĐẦU CHẤM ĐIỂM (EVALUATE) TRÊN TẬP DỮ LIỆU Y TẾ")
    print("="*60)
    
    # 1. Chuẩn bị đề thi
    test_data = generate_test_audio()
    
    # 2. Nạp Thần Y (Whisper LoRA Medical)
    model = load_model()
    
    total_audio_duration = 0.0
    total_transcribe_time = 0.0
    references = []
    hypotheses = []
    
    print("\n" + "="*60)
    print("ĐANG THỰC HIỆN BÀI THI...")
    
    for idx, item in enumerate(test_data):
        audio_path = item["path"]
        ref_text = item["reference"]
        duration = item["duration"]
        
        total_audio_duration += duration
        
        # AI Bắt đầu nghe và dịch
        start_t = time.time()
        result = transcribe(audio_path, model)
        end_t = time.time()
        
        pred_text = result["text"]
        inference_time = end_t - start_t
        total_transcribe_time += inference_time
        
        references.append(ref_text)
        hypotheses.append(pred_text)
        
        # Hiển thị trực tiếp quá trình thi
        print(f"\nCâu {idx+1}:")
        print(f"  [ĐỀ THI] : {ref_text}")
        print(f"  [AI DỊCH]: {pred_text}")
        
    # 3. Tính toán điểm số
    # Chuẩn hóa chuỗi bằng tay để tránh lỗi không tương thích phiên bản thư viện Jiwer
    transformed_refs = [r.lower().replace(".", "").replace(",", "").replace("?", "") for r in references]
    transformed_hyps = [h.lower().replace(".", "").replace(",", "").replace("?", "") for h in hypotheses]
    
    wer = jiwer.wer(transformed_refs, transformed_hyps) * 100
    cer = jiwer.cer(transformed_refs, transformed_hyps) * 100
    rtf = total_transcribe_time / total_audio_duration if total_audio_duration > 0 else 0
    
    print("\n" + "=" * 60)
    print("🏆 BẢNG ĐIỂM TỔNG KẾT (TẬP DỮ LIỆU Y TẾ) - CHỤP LẠI MÀN HÌNH NÀY")
    print("=" * 60)
    print(f"- Tổng số câu đã giải     : {len(test_data)} câu")
    print(f"- Tổng độ dài âm thanh    : {total_audio_duration:.2f} giây")
    print(f"- Tổng thời gian dịch     : {total_transcribe_time:.2f} giây")
    print(f"- Tốc độ xử lý (RTF)      : {rtf:.2f} (càng nhỏ càng tốt, < 1 là dịch nhanh hơn thời gian thực)")
    print(f"- WER (Tỷ lệ lỗi từ)      : {wer:.2f} % (Càng thấp càng tốt)")
    print(f"- CER (Tỷ lệ lỗi ký tự)   : {cer:.2f} % (Càng thấp càng tốt)")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
