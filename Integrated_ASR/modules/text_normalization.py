from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

from rapidfuzz import fuzz
from rapidfuzz.process import extractOne


# =========================================================
# 1) RULES GỐC + MỞ RỘNG (thêm nhiều biến thể thực tế hơn)
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
    "kham benh": "khám bệnh",
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
        "nhắc tôi uống thuốt": "nhắc tôi uống thuốc",
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
        "đặt lịch thứ ba tuần sau buổi sán": "đặt lịch thứ ba tuần sau buổi sáng",
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

# =========================================================
# 2) TỪ VỰNG Y TẾ CHUYÊN BIỆT (dùng cho word-level fuzzy correction)
# =========================================================
MEDICAL_VOCAB: Set[str] = {
    "đo", "huyết", "áp", "nhịp", "tim", "bệnh", "viện", "toa", "thuốc",
    "khám", "bệnh", "cấp", "cứu", "chóng", "mặt", "buồn", "nôn", "đau",
    "bụng", "thở", "nhức", "đầu", "mệt", "mỏi", "tác", "dụng", "dị",
    "ứng", "tương", "tác", "cữ", "lần", "sáng", "trưa", "tối", "ngủ",
    "hẹn", "bác", "sĩ", "lịch", "hôm", "nay", "mai", "chiều",
    # thêm các từ y tế phổ biến khác (có thể mở rộng sau)
    "huyết áp", "nhịp tim", "khám bệnh", "cấp cứu", "chóng mặt",
    "buồn nôn", "đau bụng", "khó thở", "đau đầu", "mệt mỏi", "thuốc men",
}

# =========================================================
# 3) KIỂU DỮ LIỆU
# =========================================================
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


# =========================================================
# 4) NORMALIZE + PHONETIC (xử lý accent + regional confusion)
# =========================================================
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
    """Phonetic key: bỏ dấu + map một số lỗi regional phổ biến của người già/vùng cao"""
    text = strip_accents(normalize_text(text))
    # Regional / ASR common swaps
    swaps = {
        "ch": "tr", "tr": "ch",
        "s": "x", "x": "s",
        "d": "gi", "gi": "d",
        "v": "b", "b": "v",  # một số vùng lẫn
        "l": "n", "n": "l",
        "ng": "n", "n": "ng",
        "huyet": "huyết", "ap": "áp",
        "thuoc": "thuốc", "benh": "bệnh",
        "kham": "khám", "cuu": "cứu",
    }
    for k, v in swaps.items():
        text = text.replace(k, v)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


# =========================================================
# 5) BUILD RULES (giữ nguyên nhưng dùng phonetic key)
# =========================================================
def build_rules() -> List[RuleEntry]:
    rules: List[RuleEntry] = []

    for wrong, correct in COMMON_RULES.items():
        rules.append(
            RuleEntry(
                intent="common",
                wrong=wrong,
                correct=correct,
                wrong_key=make_key(wrong),
                correct_key=make_key(correct),
                wrong_len=len(tokenize(wrong)),
            )
        )

    for intent, mapping in INTENT_RULES.items():
        for wrong, correct in mapping.items():
            rules.append(
                RuleEntry(
                    intent=intent,
                    wrong=wrong,
                    correct=correct,
                    wrong_key=make_key(wrong),
                    correct_key=make_key(correct),
                    wrong_len=len(tokenize(wrong)),
                )
            )
    return rules


ALL_RULES = build_rules()


# =========================================================
# 6) NGƯỠNG AN TOÀN (adaptive theo độ dài)
# =========================================================
def min_score_by_len(n_tokens: int) -> float:
    if n_tokens <= 1:
        return 94.0
    if n_tokens == 2:
        return 90.0
    return 86.0


def score_window_vs_rule(window_text: str, rule: RuleEntry) -> Tuple[float, float]:
    window_key = make_key(window_text)
    score_wrong = fuzz.WRatio(window_key, rule.wrong_key)
    score_correct = fuzz.WRatio(window_key, rule.correct_key)
    return score_wrong, score_correct


# =========================================================
# 7) TÌM MATCH PHRASE-LEVEL (giữ nguyên logic nhưng linh hoạt hơn)
# =========================================================
def find_phrase_candidates(text: str, rules: List[RuleEntry]) -> List[MatchResult]:
    tokens = tokenize(text)
    candidates: List[MatchResult] = []
    if not tokens:
        return candidates

    max_rule_len = max((rule.wrong_len for rule in rules), default=5)

    for start in range(len(tokens)):
        for win_len in range(1, min(max_rule_len + 2, len(tokens) - start + 1)):  # +2 để linh hoạt hơn
            end = start + win_len
            window_text = " ".join(tokens[start:end])

            for rule in rules:
                if abs(rule.wrong_len - win_len) > 2:  # nới lỏng một chút
                    continue

                score_wrong, score_correct = score_window_vs_rule(window_text, rule)

                if score_correct >= score_wrong:  # ưu tiên sửa sai
                    continue
                if (score_wrong - score_correct) < 5:  # giảm ngưỡng chênh lệch
                    continue

                threshold = min_score_by_len(rule.wrong_len)
                if score_wrong < threshold:
                    continue

                candidates.append(
                    MatchResult(
                        start=start,
                        end=end,
                        matched_text=window_text,
                        wrong=rule.wrong,
                        correct=rule.correct,
                        intent=rule.intent,
                        score_wrong=score_wrong,
                        score_correct=score_correct,
                    )
                )

    return candidates


def select_non_overlapping(candidates: List[MatchResult]) -> List[MatchResult]:
    candidates = sorted(
        candidates,
        key=lambda x: (-x.score_wrong, -(x.end - x.start), -len(x.wrong)),
    )

    chosen: List[MatchResult] = []
    occupied = set()

    for c in candidates:
        span = set(range(c.start, c.end))
        if occupied & span:
            continue
        chosen.append(c)
        occupied.update(span)

    return sorted(chosen, key=lambda x: x.start)


# =========================================================
# 8) WORD-LEVEL FUZZY CORRECTION (mới - linh hoạt nhất)
# =========================================================
def correct_single_word(token: str, vocab: Set[str], threshold: float = 88.0) -> str:
    """Sửa từng từ bằng fuzzy match với từ vựng y tế"""
    clean_token = normalize_text(token)
    if clean_token in vocab:
        return clean_token

    # Dùng phonetic key để tăng độ chính xác với accent/regional
    key = make_key(clean_token)
    best_match, score, _ = extractOne(key, [make_key(v) for v in vocab], scorer=fuzz.WRatio) or (clean_token, 0, None)

    if score >= threshold:
        # Tìm từ gốc đúng từ key
        for v in vocab:
            if make_key(v) == best_match:
                return v
    return token


def word_level_correction(tokens: List[str]) -> List[str]:
    """Pass 2: sửa từng từ"""
    corrected = []
    for token in tokens:
        corrected.append(correct_single_word(token, MEDICAL_VOCAB))
    return corrected


# =========================================================
# 9) POST PROCESS CHÍNH (TWO-PASS: phrase → word)
# =========================================================
def post_process_text_auto(text: str) -> Tuple[str, List[MatchResult], Optional[str]]:
    norm_text = normalize_text(text)
    tokens = tokenize(norm_text)

    # PASS 1: Phrase-level (intent-aware, rất chính xác cho cụm dài)
    candidates = find_phrase_candidates(norm_text, ALL_RULES)
    phrase_matches = select_non_overlapping(candidates)

    # Áp dụng sửa phrase
    rebuilt: List[str] = []
    cursor = 0
    for m in phrase_matches:
        rebuilt.extend(tokens[cursor : m.start])
        rebuilt.extend(tokenize(m.correct))
        cursor = m.end
    rebuilt.extend(tokens[cursor:])

    # PASS 2: Word-level fuzzy trên kết quả sau phrase
    final_tokens = word_level_correction(rebuilt)
    clean_text = " ".join(final_tokens)

    # Tính intent
    intent_counter = Counter(m.intent for m in phrase_matches if m.intent != "common")
    likely_intent = intent_counter.most_common(1)[0][0] if intent_counter else None

    return clean_text, phrase_matches, likely_intent


# =========================================================
# 10) CLI TEST (cập nhật thông báo)
# =========================================================
def interactive_cli() -> None:
    print("=" * 90)
    print("🚀 MEDICAL ASR POST-PROCESSING – PHIÊN BẢN TỐI ƯU (Hybrid)")
    print("• Phrase-level rules (intent-aware)")
    print("• Word-level fuzzy + phonetic normalization")
    print("• Dành cho người già / vùng cao (accent, missing tone, regional error)")
    print("Lệnh: /exit để thoát")
    print("=" * 90)

    while True:
        raw = input("\nNhập câu nói (có lỗi): ").strip()
        if not raw:
            continue
        if raw.lower() == "/exit":
            print("✅ Thoát.")
            break

        clean, matches, likely_intent = post_process_text_auto(raw)

        print("\n📌 KẾT QUẢ")
        print(f"  Raw          : {raw}")
        print(f"  Clean        : {clean}")
        print(f"  Likely intent: {likely_intent or 'unknown'}")

        if matches:
            print("  Các cụm phrase được sửa:")
            for i, m in enumerate(matches, 1):
                print(
                    f"    {i}. '{m.matched_text}' → '{m.correct}' "
                    f"| intent={m.intent} "
                    f"| score={m.score_wrong:.1f}"
                )
        else:
            print("  ✅ Không cần sửa phrase nào – chỉ word-level fuzzy.")


if __name__ == "__main__":
    interactive_cli()