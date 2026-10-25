from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

from rapidfuzz import fuzz
from rapidfuzz.process import extractOne


# =========================================================
# 1) RULES & VOCAB (giữ nguyên như cũ)
# =========================================================
COMMON_RULES: Dict[str, str] = {
    "đỏ mép áo": "đo huyết áp",
    "đô huyết áp": "đo huyết áp",
    "do huyet ap": "đo huyết áp",
    "huyết app": "huyết áp",
    "nhịp tìm": "nhịp tim",
    "bịnh viện": "bệnh viện",
    "toa thuốt": "toa thuốc",
    "kham benh": "khám bệnh",
    "thuoc": "thuốc",
    "thuoc men": "thuốc men",
    "benh vien": "bệnh viện",
    "cap cuu": "cấp cứu",
    "huyet ap": "huyết áp",
}

INTENT_RULES: Dict[str, Dict[str, str]] = {
    "request_emergency": {
        "gọi cáp cứu": "gọi cấp cứu",
        "gọi cắp cứu": "gọi cấp cứu",
        "tôi cần cáp cứu": "tôi cần cấp cứu",
        "khó thở quá chời": "khó thở quá",
        "đau ngực giữ dội": "đau ngực dữ dội",
        "tui bị ngất xỉu": "tôi bị ngất xỉu",
        "đưa tui đi viện gấp": "đưa tôi đi viện gấp",
        "ngat xiêu": "ngất xỉu",
        "khó thở nặng": "khó thở",
    },
    "report_symptom": {
        "tôi bị tróng mặt": "tôi bị chóng mặt",
        "tôi bị bùn nôn": "tôi bị buồn nôn",
        "tôi bị buồn lôn": "tôi bị buồn nôn",
        "đau bung âm ỉ": "đau bụng âm ỉ",
        "khó thỡ": "khó thở",
        "đau nhứt đầu": "đau nhức đầu",
        "mệt mõi": "mệt mỏi",
        "chóng mặt": "chóng mặt",
        "buồn nôn": "buồn nôn",
        "đau bụng": "đau bụng",
        "mỏi mệt": "mệt mỏi",
    },
    "inquire_medicine": {
        "thuốc này uống sao": "thuốc này uống như thế nào",
        "thuốc này uống như răng": "thuốc này uống như thế nào",
        "thuốc này có tác dung gì": "thuốc này có tác dụng gì",
        "thuốc này có tát dụng gì": "thuốc này có tác dụng gì",
        "thuốc này có bị ứng không": "thuốc này có bị dị ứng không",
        "thuốc này có tương tát không": "thuốc này có tương tác không",
        "thuốc này uống mấy cử": "thuốc này uống mấy cữ",
    },
    "set_or_update_medication": {
        "nhắc tui uống thuốt": "nhắc tôi uống thuốc",
        "đặt lịch uống thuốt": "đặt lịch uống thuốc",
        "uống ngày hai cử": "uống ngày hai cữ",
        "mỗi ngày hai lân": "mỗi ngày hai lần",
        "sau khi ăn sang": "sau khi ăn sáng",
        "trước khi đi ngũ uống thuốc": "trước khi đi ngủ uống thuốc",
    },
    "cancel_medication": {
        "hủy nhắc thuốc men": "hủy nhắc thuốc",
        "thôi nhắc uống thuốt": "thôi nhắc uống thuốc",
        "dừng thuốc men chiều nay": "dừng thuốc chiều nay",
        "hủy liệt uống thuốc": "hủy lịch uống thuốc",
        "ngưng nhắc thuốt": "ngưng nhắc thuốc",
        "khỏi cần nhắc thuốc nữa nhe": "khỏi cần nhắc thuốc nữa nhé",
        "bỏ cử thuốc tối": "bỏ cữ thuốc tối",
    },
    "check_medication": {
        "bữa nay tôi có thuốc gì": "hôm nay tôi có thuốc gì",
        "coi giùm thuốc hôm nay": "xem giúp thuốc hôm nay",
        "kiểm tra thuốc dùm": "kiểm tra thuốc giùm",
        "thuốc chiều nay uống chưa": "thuốc chiều nay đã uống chưa",
        "tối nay có thuốc không ta": "tối nay có thuốc không",
        "toa thuốc bữa nay": "toa thuốc hôm nay",
        "còn cử thuốc nào chưa": "còn cữ thuốc nào chưa",
    },
    "set_or_update_appointment": {
        "đặt lịch khắm": "đặt lịch khám",
        "đặt nịch khám": "đặt lịch khám",
        "hẹn bác sỹ": "hẹn bác sĩ",
        "đặt lịch gặp bát sĩ": "đặt lịch gặp bác sĩ",
        "hẹn đi kham benh": "hẹn đi khám bệnh",
        "đổi lịch khắm": "đổi lịch khám",
    },
    "cancel_appointment": {
        "hủy lịch khắm": "hủy lịch khám",
        "thôi không đi khắm nữa": "thôi không đi khám nữa",
        "bỏ lịch hẹn bác sỹ": "bỏ lịch hẹn bác sĩ",
        "hủy cái hẹn ngày mai": "hủy cái hẹn ngày mai",
        "dẹp lịch khắm": "hủy lịch khám",
        "khỏi đặt lịch nữa": "hủy lịch hẹn",
        "bỏ buổi khắm chiều nay": "bỏ buổi khám chiều nay",
    },
    "check_appointment": {
        "coi lịch hẹn khám": "xem lịch hẹn khám",
        "kiểm tra nịch hẹn": "kiểm tra lịch hẹn",
        "mai có hẹn khắm không": "mai có hẹn khám không",
        "bữa nay có lịch bác sỹ không": "hôm nay có lịch bác sĩ không",
        "lịch hẹn của tui đâu": "lịch hẹn của tôi đâu",
        "tôi hẹn khắm lúc mấy giờ": "tôi hẹn khám lúc mấy giờ",
        "xem giúp cái hẹn đi viện": "xem giúp lịch hẹn đi viện",
    },
}

MEDICAL_VOCAB: Set[str] = {
    "đo", "huyết", "áp", "nhịp", "tim", "bệnh", "viện", "toa", "thuốc",
    "khám", "bệnh", "cấp", "cứu", "chóng", "mặt", "buồn", "nôn", "đau",
    "bụng", "thở", "nhức", "đầu", "mệt", "mỏi", "tác", "dụng", "dị",
    "ứng", "tương", "tác", "cữ", "lần", "sáng", "trưa", "tối", "ngủ",
    "hẹn", "bác", "sĩ", "lịch", "hôm", "nay", "mai", "chiều",
    "huyết áp", "nhịp tim", "khám bệnh", "cấp cứu", "chóng mặt",
    "buồn nôn", "đau bụng", "khó thở", "đau đầu", "mệt mỏi", "thuốc men",
}


# =========================================================
# Các hàm còn lại giữ nguyên (normalize, make_key, tokenize, build_rules, ...)
# =========================================================
# ... (tôi giữ nguyên phần này để code ngắn gọn, bạn copy nguyên từ code cũ của bạn)

@dataclass(frozen=True)
class RuleEntry:
    intent: str
    wrong: str
    correct: str
    wrong_key: str
    correct_key: str
    wrong_len: int


@dataclass
class MatchResult:
    start: int
    end: int
    matched_text: str
    wrong: str
    correct: str
    intent: str
    score_wrong: float
    score_correct: float


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def make_key(text: str) -> str:
    text = strip_accents(normalize_text(text))
    swaps = {
        "ch": "tr", "tr": "ch", "s": "x", "x": "s", "d": "gi", "gi": "d",
        "v": "b", "b": "v", "l": "n", "n": "l", "ng": "n", "n": "ng",
        "huyet": "huyết", "ap": "áp", "thuoc": "thuốc", "benh": "bệnh",
        "kham": "khám", "cuu": "cứu",
    }
    for k, v in swaps.items():
        text = text.replace(k, v)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def build_rules() -> List[RuleEntry]:
    rules: List[RuleEntry] = []
    for wrong, correct in COMMON_RULES.items():
        rules.append(RuleEntry("common", wrong, correct, make_key(wrong), make_key(correct), len(tokenize(wrong))))
    for intent, mapping in INTENT_RULES.items():
        for wrong, correct in mapping.items():
            rules.append(RuleEntry(intent, wrong, correct, make_key(wrong), make_key(correct), len(tokenize(wrong))))
    return rules


ALL_RULES = build_rules()


def min_score_by_len(n_tokens: int) -> float:
    if n_tokens <= 1: return 94.0
    if n_tokens == 2: return 90.0
    return 86.0


def score_window_vs_rule(window_text: str, rule: RuleEntry) -> Tuple[float, float]:
    window_key = make_key(window_text)
    return fuzz.WRatio(window_key, rule.wrong_key), fuzz.WRatio(window_key, rule.correct_key)


def find_phrase_candidates(text: str, rules: List[RuleEntry]) -> List[MatchResult]:
    tokens = tokenize(text)
    candidates: List[MatchResult] = []
    if not tokens:
        return candidates

    max_rule_len = max((r.wrong_len for r in rules), default=5)

    for start in range(len(tokens)):
        for win_len in range(1, min(max_rule_len + 2, len(tokens) - start + 1)):
            end = start + win_len
            window_text = " ".join(tokens[start:end])

            for rule in rules:
                if abs(rule.wrong_len - win_len) > 2:
                    continue
                score_wrong, score_correct = score_window_vs_rule(window_text, rule)
                if score_correct >= score_wrong or (score_wrong - score_correct) < 5:
                    continue
                if score_wrong < min_score_by_len(rule.wrong_len):
                    continue

                candidates.append(MatchResult(start, end, window_text, rule.wrong, rule.correct, rule.intent, score_wrong, score_correct))
    return candidates


def select_non_overlapping(candidates: List[MatchResult]) -> List[MatchResult]:
    candidates = sorted(candidates, key=lambda x: (-x.score_wrong, -(x.end - x.start), -len(x.wrong)))
    chosen: List[MatchResult] = []
    occupied = set()
    for c in candidates:
        span = set(range(c.start, c.end))
        if occupied & span:
            continue
        chosen.append(c)
        occupied.update(span)
    return sorted(chosen, key=lambda x: x.start)


def correct_single_word(token: str, vocab: Set[str], threshold: float = 88.0) -> str:
    clean_token = normalize_text(token)
    if clean_token in vocab:
        return clean_token

    key = make_key(clean_token)
    vocab_keys = [make_key(v) for v in vocab]
    result = extractOne(key, vocab_keys, scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        best_key = result[0]
        for v in vocab:
            if make_key(v) == best_key:
                return v
    return token


def word_level_correction(tokens: List[str]) -> List[str]:
    return [correct_single_word(t, MEDICAL_VOCAB) for t in tokens]


# =========================================================
# 9) POST PROCESS CHÍNH
# =========================================================
def post_process_text_auto(text: str) -> Tuple[str, List[MatchResult], Optional[str]]:
    norm_text = normalize_text(text)
    tokens = tokenize(norm_text)

    candidates = find_phrase_candidates(norm_text, ALL_RULES)
    phrase_matches = select_non_overlapping(candidates)

    # Áp dụng sửa phrase
    rebuilt: List[str] = []
    cursor = 0
    for m in phrase_matches:
        rebuilt.extend(tokens[cursor:m.start])
        rebuilt.extend(tokenize(m.correct))
        cursor = m.end
    rebuilt.extend(tokens[cursor:])

    # Word-level correction
    final_tokens = word_level_correction(rebuilt)
    clean_text = " ".join(final_tokens)

    intent_counter = Counter(m.intent for m in phrase_matches if m.intent != "common")
    likely_intent = intent_counter.most_common(1)[0][0] if intent_counter else None

    return clean_text, phrase_matches, likely_intent


# =========================================================
# 10) MAIN - ĐỌC TỪ FILE output_text_from_voice.txt
# =========================================================
def main():
    input_file = "output_text_from_voice.txt"   # ← thay tên file nếu khác

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        
        if not raw_text:
            print("File rỗng hoặc không có nội dung!")
            return

        print("=== ĐANG XỬ LÝ TEXT TỪ GIỌNG NÓI ===")
        print(f"Raw text     : {raw_text}")

        clean_text, matches, likely_intent = post_process_text_auto(raw_text)

        print(f"Clean text   : {clean_text}")
        print(f"Likely intent: {likely_intent or 'unknown'}")

        if matches:
            print("\nCác cụm đã được sửa (phrase level):")
            for i, m in enumerate(matches, 1):
                print(f"  {i}. '{m.matched_text}' → '{m.correct}'  (intent: {m.intent})")

        # Lưu kết quả sạch ra file mới
        output_file = "cleaned_medical_text.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(clean_text)
        
        print(f"\nĐã lưu kết quả sạch vào file: {output_file}")

    except FileNotFoundError:
        print(f"Không tìm thấy file: {input_file}")
        print("Vui lòng đặt file 'output_text_from_voice.txt' cùng thư mục với file python này.")
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()