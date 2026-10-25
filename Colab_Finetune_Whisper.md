# HƯỚNG DẪN HUẤN LUYỆN AI Y TẾ TRÊN GOOGLE COLAB (VERSION 2.0)

> **QUAN TRỌNG:** Bạn hãy nén thư mục `D:\Medical_Audio_Dataset` thành file **`Medical_Audio_Dataset.rar`** (hoặc zip) rồi tải lên thư mục `HuanLuyenAI_NCKH` trên Google Drive nhé.

Dưới đây là 5 Ô Code đã được tôi tối ưu lại dành riêng cho 2000 câu y tế. Bạn hãy xóa hết code cũ trên Colab và dán lần lượt 5 ô này vào:

### Ô Code 1: Giải nén Dữ liệu Y tế
```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/dataset
# Giải nén file dữ liệu mới
!unrar x /content/drive/MyDrive/HuanLuyenAI_NCKH/Medical_Audio_Dataset.rar /content/dataset/

print("Giải nén hoàn tất!")
```

### Ô Code 2: Cài đặt Vũ khí hạng nặng
*(Nếu máy chưa cài thì chạy, nếu cài rồi chạy lại vẫn không sao)*
```bash
!pip install -q transformers datasets peft accelerate evaluate jiwer librosa bitsandbytes soundfile torchao
```

### Ô Code 3: Nạp Dữ liệu Y tế (2000 câu)
```python
import os
from datasets import load_dataset, Audio
from transformers import WhisperProcessor

print("Đang nạp dữ liệu...")
dataset = load_dataset("audiofolder", data_dir="/content/dataset/Medical_Audio_Dataset", drop_labels=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="vi", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Đang chuyển đổi dữ liệu thành sóng âm...")
# Sử dụng toàn bộ 2000 mẫu Y tế
train_dataset = dataset["train"].map(prepare_dataset, num_proc=2)
print("Hoàn tất xử lý dữ liệu!")
```

### Ô Code 4: Thiết lập Kiến trúc LoRA
```python
import torch
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

print("Đang tải não bộ Whisper Base...")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-base", 
    device_map="auto"
)
model.requires_grad_(False)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

### Ô Code 5: BẤM NÚT HUẤN LUYỆN
*(Lần này AI sẽ học nghiêm túc hơn với 800 bước. Bạn cắm sạc laptop và đi uống tách cafe nhé!)*
```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/Whisper_LoRA_Medical",
    per_device_train_batch_size=8,
    learning_rate=1e-3,
    max_steps=800, # Tăng số bước học để ngấm kỹ 2000 câu y tế
    logging_steps=50,
    save_steps=200,
    fp16=True, 
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("🚀 CHÍNH THỨC BẮT ĐẦU HUẤN LUYỆN Y TẾ...")
trainer.train()

# Lưu kết quả về Google Drive dưới tên mới
model.save_pretrained("/content/drive/MyDrive/Whisper_LoRA_Medical_Final")
print("🎉 ĐÃ LƯU MÔ HÌNH THẦN Y THÀNH CÔNG VÀO GOOGLE DRIVE!")
```
