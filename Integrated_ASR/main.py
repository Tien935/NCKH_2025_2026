import os
import time
import imageio_ffmpeg
import shutil
import glob

# Thêm ffmpeg vào PATH của môi trường để Whisper có thể tìm thấy (Whisper gọi cứng tên "ffmpeg")
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
local_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
if not os.path.exists(local_ffmpeg):
    shutil.copy(ffmpeg_exe, local_ffmpeg)
os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ.get("PATH", "")

from modules.vad import remove_silence
from modules.asr_whisper import load_model, transcribe
from modules.text_normalization import post_process_text_auto
from modules.nlu_classifier import predict_intent

def main():
    print("=" * 60)
    print("🚀 TÍCH HỢP HỆ THỐNG ASR VÀ NLP CHO Y TẾ (CHẾ ĐỘ HÀNG LOẠT)")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_dir = os.path.join(base_dir, "data", "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    
    # Tìm tất cả các file âm thanh trong thư mục data/inputs/
    audio_extensions = ["*.m4a", "*.wav", "*.mp3"]
    input_files = []
    for ext in audio_extensions:
        input_files.extend(glob.glob(os.path.join(inputs_dir, ext)))

    if not input_files:
        print(f"Lỗi: Không tìm thấy file âm thanh nào.")
        print(f"Vui lòng copy các file cần test (.m4a, .wav, .mp3) vào thư mục: {inputs_dir}")
        return

    print(f"📁 Đã tìm thấy {len(input_files)} file cần xử lý. Bắt đầu nạp Mô hình AI...")
    
    # Tải mô hình ASR 1 lần duy nhất để tiết kiệm thời gian
    try:
        model = load_model()
        print("✅ Đã nạp thành công bộ não Thần Y (Whisper LoRA).")
    except Exception as e:
        print("❌ Lỗi khi nạp mô hình ASR:", e)
        return

    # Duyệt qua từng file
    for idx, input_audio in enumerate(input_files, 1):
        filename = os.path.basename(input_audio)
        vad_audio = os.path.join(base_dir, "data", f"vad_temp_{idx}.wav")
        
        print("\n" + "=" * 60)
        print(f"🎧 FILE {idx}/{len(input_files)}: {filename}")
        print("=" * 60)
        
        try:
            # 1. Xử lý VAD
            remove_silence(input_audio, vad_audio)
            
            # 2. Nhận dạng giọng nói (ASR)
            result = transcribe(vad_audio, model)
            raw_text = result["text"]
            
            # 3. Chuẩn hóa Văn bản (NLP)
            clean_text, matches, old_intent = post_process_text_auto(raw_text)
            
            # 4. Phân loại Intent (NLU)
            dl_intent, confidence = predict_intent(clean_text)
            
            # IN KẾT QUẢ
            print(f"📝 Raw Text (ASR)  : {raw_text}")
            print(f"✨ Clean Text (NLP): {clean_text}")
            print(f"🧠 Intent (BiLSTM) : {dl_intent}")
            print(f"📊 Độ tự tin (Conf): {confidence*100:.2f}%")
            
            # Dọn dẹp file tạm
            if os.path.exists(vad_audio):
                os.remove(vad_audio)
                
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {filename}:", e)

if __name__ == "__main__":
    main()
