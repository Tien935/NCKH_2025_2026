import whisper
import time
import os
import subprocess
from Text_normalization import post_process_text_auto

# ===== CONFIG =====
MODEL_SIZE = "medium"   # tiny | base | small | medium | large
INPUT_RAW = "audio_processing/input.m4a"
INPUT_VAD = "output.wav"


# ===== LOAD MODEL =====
def load_model():
    print("Đang load Whisper model...")
    model = whisper.load_model(MODEL_SIZE)
    return model


# ===== CONVERT AUDIO (FFMPEG) =====
def convert_to_wav(input_file, output_file="temp.wav"):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ac", "1",
        "-ar", "16000",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(output_file):
        raise Exception("Convert thất bại!")

    return output_file


# ===== TRANSCRIBE =====
def transcribe(model, file_path):
    print(f"\nĐang xử lý: {file_path}")

    if not os.path.exists(file_path):
        raise Exception(f"Không tìm thấy file: {file_path}")

    start_time = time.time()

    result = model.transcribe(
        file_path,
        language="vi",   # ép tiếng Việt
        task = "transcribe" #tránh dịch sang tiếng anh
    )

    end_time = time.time()

    latency = (end_time - start_time) * 1000  # ms
    text = result["text"]

    return text.strip(), latency


# ===== MAIN =====
def main():
    model = load_model()

    # ===== FILE GỐC =====
    try:
        raw_wav = convert_to_wav(INPUT_RAW)
        text1, time1 = transcribe(model, raw_wav)
    except Exception as e:
        print("Lỗi file gốc:", e)
        text1, time1 = "", 0

    # ===== FILE SAU VAD =====
    text2, time2 = transcribe(model, INPUT_VAD)

    # ===== REPORT =====
    print("\n================= REPORT =================")

    if time1 > 0:
        print("\nFILE GỐC:")
        print("Latency:", round(time1, 2), "ms")
        print("Text:", text1)

    print("\nFILE SAU VAD:")
    print("Latency:", round(time2, 2), "ms")
    print("Text:", text2)

    if time1 > 0:
        improvement = ((time1 - time2) / time1) * 100
        print("\nHIỆU NĂNG:")
        print(f"Giảm thời gian: {round(improvement, 2)} %")

    print("\nDone!")


# ===== RUN =====
if __name__ == "__main__":
    main()