import torch
import numpy as np
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Cấu hình đường dẫn cố định
LORA_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Whisper_LoRA_Medical_Final")

class WhisperLoRAWrapper:
    """
    Lớp bao bọc (Wrapper) để giả lập lại cách gọi hàm model.transcribe() của thư viện openai-whisper cũ,
    giúp cho các file main.py và evaluate_asr.py KHÔNG cần phải sửa đổi bất kỳ dòng code nào.
    """
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
    def transcribe(self, audio_data, **kwargs):
        # 1. Xử lý đầu vào (string path hoặc numpy array)
        if isinstance(audio_data, str):
            import soundfile as sf
            audio_array, sampling_rate = sf.read(audio_data)
            if sampling_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        else:
            audio_array = audio_data
            
        # Đảm bảo định dạng chuẩn float32
        audio_array = np.array(audio_array, dtype=np.float32)
            
        # 2. Rút trích đặc trưng âm thanh
        input_features = self.processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features 
        
        # 3. Ép AI dịch tiếng Việt
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        
        # 4. Dự đoán (Inference)
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            
        # 5. Giải mã ID ra văn bản
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Trả về kết quả dưới dạng Dictionary y hệt bản gốc
        return {"text": transcription.strip()}

def load_model():
    print("="*60)
    print("🧠 ĐANG NẠP MÔ HÌNH NHẬN DIỆN GIỌNG NÓI")
    
    # Bảo vệ ổ C
    os.environ["HF_HOME"] = "D:\\HF_Cache"
    
    print("1. Đang tải kiến trúc Whisper Base...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="vi", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    if os.path.exists(LORA_MODEL_DIR):
        print(f"2. ✅ Đã tìm thấy Não bộ LoRA (Fine-tuned) tại:\n   {LORA_MODEL_DIR}")
        print("3. Đang tiến hành cấy ghép mạch LoRA vào Whisper Base...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
        print("✅ Cấy ghép thành công! AI của bạn đã thông minh hơn.")
    else:
        print(f"2. ⚠️ Không tìm thấy thư mục LoRA tại:\n   {LORA_MODEL_DIR}")
        print("   Hệ thống sẽ chạy tạm bằng mô hình Whisper Base gốc (chưa độ)!")
        model = base_model
        
    model.eval()
    print("="*60)
    
    # Bọc model lại để giả lập giao diện API cũ
    return WhisperLoRAWrapper(model, processor)

def transcribe(audio_path, model):
    """ Hàm này giữ nguyên để tương thích với cấu trúc cũ """
    return model.transcribe(audio_path)