import torch
import subprocess
import os
import wave
import numpy as np

# ===== LOAD MODEL SILERO =====
torch.hub.set_dir("D:\\HF_Cache\\torch")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)

(get_speech_timestamps,
 _, _, _, collect_chunks) = utils


import imageio_ffmpeg

# ===== CONVERT BẰNG FFMPEG =====
def convert_to_wav(input_file, output_file="temp.wav"):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-y",
        "-i", input_file,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_file
    ]
    result = subprocess.run(command)

    if result.returncode != 0 or not os.path.exists(output_file):
        raise Exception("Convert thất bại! Kiểm tra ffmpeg hoặc file input.")

    return output_file


# ===== ĐỌC WAV (KHÔNG DÙNG TORCHAUDIO) =====
def read_wave(path):
    with wave.open(path, 'rb') as wf:
        sample_rate = wf.getframerate()
        audio = wf.readframes(wf.getnframes())

    # convert bytes → numpy → tensor
    audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio_np), sample_rate


# ===== GHI WAV =====
def write_wave(path, audio_tensor, sample_rate):
    audio_np = (audio_tensor.numpy() * 32768).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_np.tobytes())


# ===== MAIN =====
def remove_silence(input_file, output_file):
    print("Đang convert bằng ffmpeg...")
    temp_wav = convert_to_wav(input_file)

    print("Đang đọc audio...")
    wav, sr = read_wave(temp_wav)

    print("Đang detect voice...")
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=0.5  # chỉnh độ nhạy
    )

    print("Đang cắt silence...")
    processed_audio = collect_chunks(speech_timestamps, wav)

    print("Đang lưu file...")
    write_wave(output_file, processed_audio, sr)

    os.remove(temp_wav)

    print("Cắt xong rồi!")

