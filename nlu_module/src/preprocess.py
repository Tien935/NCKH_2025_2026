import pandas as pd
import json
import os
import re

# =========================
# PATHS
# =========================
CONFIG_PATH = "data/config/intents.json"
RAW_DATA_PATH = "data/raw/mock_data.xlsx"
PROCESSED_DATA_PATH = "data/processed/cleaned_mock_data.csv"


# =========================
# LOAD 18 INTENTS
# =========================
def load_intents(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return set(json.load(f))


# =========================
# PREPROCESS
# =========================
def preprocess_data():

    valid_intents = load_intents(CONFIG_PATH)

    # đọc excel KHÔNG lấy header
    raw = pd.read_excel(
        RAW_DATA_PATH,
        header=None,
        usecols="A:C"
    )

    raw.columns = ["sentence", "intent", "type"]

    print("Preview raw data:")
    print(raw.head(15))


    cleaned_rows = []

    for _, row in raw.iterrows():

        sentence = (
            str(row["sentence"]).strip()
            if pd.notna(row["sentence"])
            else ""
        )

        intent = (
            str(row["intent"]).strip()
            if pd.notna(row["intent"])
            else ""
        )

        typ = (
            str(row["type"]).strip()
            if pd.notna(row["type"])
            else ""
        )


        # -----------------
        # REMOVE NOISE
        # -----------------

        # dòng trống
        if sentence == "":
            continue

        # bỏ:
        # 1. request_emergency
        if re.match(r'^\d+\.\s', sentence):
            continue

        # bỏ header lặp
        if sentence.lower() == "sentence":
            continue


        # normalize text
        sentence = sentence.lower().strip()


        # -----------------
        # VALIDATE INTENT
        # -----------------
        if intent not in valid_intents:
            intent = "fallback"


        cleaned_rows.append(
            {
                "sentence": sentence,
                "intent": intent,
                "type": typ
            }
        )


    df = pd.DataFrame(cleaned_rows)

    # bỏ duplicate nếu có
    df = df.drop_duplicates(
        subset=["sentence", "intent"]
    ).reset_index(drop=True)


    # -----------------
    # SAVE
    # -----------------
    os.makedirs(
        os.path.dirname(PROCESSED_DATA_PATH),
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        encoding="utf-8"
    )


    print("\n========== DONE ==========")
    print(f"Số mẫu sau làm sạch: {len(df)}")
    print("\nPhân bố intent:")
    print(df["intent"].value_counts())

    print(f"\nSaved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()