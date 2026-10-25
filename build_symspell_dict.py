import os
from collections import Counter

PREFIXES = ["trợ lý ơi", "cháu ơi", "con ơi", "alo", "", "bạn ơi", "này", "nghe này", "bác sĩ ơi", "cô ơi"]
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
SUFFIXES = ["ngay lập tức", "đi", "nhé", "nhanh lên", "", "giúp tôi", "với nhé", "khẩn cấp", "một chút", "lẹ lên"]

# Bổ sung tần suất cho các từ cực kỳ thông dụng để SymSpell không bao giờ sửa chúng
EXTRA_WORDS = ["tôi", "bị", "bệnh", "đau", "ốm", "là", "thì", "mà", "ở", "của", "và", "nhưng", "bác", "sĩ", "thuốc", "uống"]

unigrams = Counter()
bigrams = Counter()

# Quét qua toàn bộ 4000 tổ hợp câu Y tế để đếm tần suất ngữ cảnh
for p in PREFIXES:
    for c in CORES:
        for s in SUFFIXES:
            sentence = f"{p} {c} {s}".strip()
            sentence = " ".join(sentence.split()) # chuẩn hóa dấu cách
            if not sentence: continue
            
            words = sentence.split()
            for w in words:
                unigrams[w] += 10 # Tăng trọng số
                
            for i in range(len(words)-1):
                bg = f"{words[i]} {words[i+1]}"
                bigrams[bg] += 10

for w in EXTRA_WORDS:
    unigrams[w] += 50000 # Cho các đại từ trọng số khổng lồ

dict_dir = os.path.join(os.path.dirname(__file__), "Integrated_ASR", "data")
os.makedirs(dict_dir, exist_ok=True)

with open(os.path.join(dict_dir, "unigram_dict.txt"), "w", encoding="utf-8") as f:
    for w, c in unigrams.most_common():
        f.write(f"{w} {c}\n")

with open(os.path.join(dict_dir, "bigram_dict.txt"), "w", encoding="utf-8") as f:
    for bg, c in bigrams.most_common():
        f.write(f"{bg} {c}\n")

print("✅ Đã tạo xong Từ điển Context cho SymSpell!")
