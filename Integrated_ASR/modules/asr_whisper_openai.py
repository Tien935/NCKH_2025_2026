import whisper
import os

# Cấu hình thư mục cache về ổ D
os.environ["XDG_CACHE_HOME"] = "D:\\HF_Cache"
os.environ["WHISPER_CACHE"] = "D:\\HF_Cache\\whisper"

def load_model():
    print("Đang nạp mô hình Whisper Base (Gốc)...")
    # Khởi tạo Whisper Base (chạy trên CPU do không có GPU tương thích PyTorch)
    model = whisper.load_model("base", download_root="D:\\HF_Cache\\whisper")
    return model

def transcribe_audio(audio_path, model):
    """
    audio_path có thể là đường dẫn file âm thanh (.wav) hoặc Numpy Array (chứa sóng âm thanh).
    Hàm model.transcribe() của OpenAI Whisper hỗ trợ cả hai.
    """
    result = model.transcribe(audio_path, language="vi")
    return result
