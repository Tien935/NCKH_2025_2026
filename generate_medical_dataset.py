import os
import csv
import random
import subprocess
from gtts import gTTS

# Cấu hình thư mục lưu trữ
OUTPUT_DIR = "D:\\Medical_Audio_Dataset"
WAVS_DIR = os.path.join(OUTPUT_DIR, "wavs")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(WAVS_DIR, exist_ok=True)

# Khởi tạo từ vựng
PREFIXES = ["Trợ lý ơi, ", "Cháu ơi, ", "Con ơi, ", "Alo, ", "", "Bạn ơi, ", "Này, ", "Nghe này, ", "Bác sĩ ơi, ", "Cô ơi, "]
CORES = [
    "tôi đau đầu quá", "hãy gọi cấp cứu cho tôi", "tôi thấy chóng mặt và buồn nôn",
    "huyết áp của tôi tự nhiên tăng cao", "lấy cho tôi vỉ thuốc trợ tim",
    "tôi bị khó thở quá", "gọi bác sĩ giúp tôi với", "ngực tôi đau thắt lại",
    "tôi bị ngã đau quá", "cứu tôi với", "đo nhịp tim giúp tôi",
    "đo huyết áp cho tôi", "nhắc tôi uống thuốc", "tôi bị chóng mặt",
    "hướng dẫn tôi cách sơ cứu", "tôi cần đi bệnh viện", "đầu tôi đau như búa bổ",
    "tôi bị ho và sốt", "tôi thấy rất mệt", "tôi bị trúng gió rồi",
    "gọi người nhà cho tôi", "tôi buồn nôn quá", "tôi bị đau bụng dữ dội",
    "tôi không thở được", "tim tôi đập nhanh quá", "tôi bị lạnh run người",
    "tôi bị dị ứng rồi", "tôi bị ong đốt", "tôi bị đứt tay chảy máu",
    "mang cho tôi ly nước ấm", "tôi bị tê tay chân", "tôi không đứng lên được",
    "gọi xe cứu thương ngay", "tôi bị đau lưng quá", "mắt tôi mờ đi không thấy rõ",
    "tôi bị ngạt mũi", "tôi thấy nhức mỏi toàn thân", "tôi bị mất ngủ",
    "lấy cho tôi máy đo huyết áp", "tôi cần uống thuốc giảm đau"
]
SUFFIXES = [" ngay lập tức", " đi", " nhé", " nhanh lên", "", " giúp tôi", " với nhé", " khẩn cấp", " một chút", " lẹ lên"]

def generate_audio(text, mp3_path):
    # Dùng Google TTS đọc văn bản (Giọng Việt Nam chuẩn)
    tts = gTTS(text, lang='vi', slow=False)
    tts.save(mp3_path)

def convert_to_wav_slow(mp3_path, wav_path):
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Chọn ngẫu nhiên tỷ lệ làm chậm (Từ 0.75 đến 0.9) để giả giọng người ốm yếu thều thào
    tempo = round(random.uniform(0.75, 0.9), 2)
    
    # Chuyển mp3 sang wav 16kHz mono và ép giảm tốc độ bằng filter atempo của FFmpeg
    cmd = [
        ffmpeg_exe, "-y", "-i", mp3_path,
        "-filter:a", f"atempo={tempo}",
        "-ac", "1", "-ar", "16000", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(mp3_path) # Xóa file mp3 tạm

def main():
    print("="*60)
    print("🚑 KHỞI ĐỘNG XƯỞNG SẢN XUẤT DỮ LIỆU Y TẾ (GOOGLE TTS ENGINE)")
    print("="*60)
    
    # Sinh ngẫu nhiên 2000 câu không trùng lặp
    sentences = set()
    while len(sentences) < 2000:
        p = random.choice(PREFIXES)
        c = random.choice(CORES)
        s = random.choice(SUFFIXES)
        sentences.add(f"{p}{c}{s}".strip().lower())
        
    sentences = list(sentences)
    
    with open(METADATA_PATH, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "transcription"])
        
        for i, text in enumerate(sentences):
            file_name = f"medical_{i:04d}.wav"
            mp3_path = os.path.join(WAVS_DIR, f"temp_{i}.mp3")
            wav_path = os.path.join(WAVS_DIR, file_name)
            
            try:
                # Bước 1: Gọi Google AI đọc
                generate_audio(text, mp3_path)
                
                # Bước 2: Ép định dạng 16kHz & Làm méo giọng cho giống người ốm
                convert_to_wav_slow(mp3_path, wav_path)
                
                # Bước 3: Ghi sổ
                writer.writerow([f"wavs/{file_name}", text])
            except Exception as e:
                print(f"Bỏ qua câu {i} do lỗi mạng: {e}")
                continue
            
            if (i+1) % 50 == 0:
                print(f"[*] Đã sản xuất: {i+1}/2000 file...")

    print("="*60)
    print("✅ ĐÃ CHẾ TẠO XONG 2000 MẪU DỮ LIỆU Y TẾ CỦA NGƯỜI GIÀ!")
    print(f"👉 Thư mục lưu trữ: {OUTPUT_DIR}")
    print("👉 Hãy nén thành file ZIP và upload lên Google Drive như cũ!")
    print("="*60)

if __name__ == "__main__":
    main()
