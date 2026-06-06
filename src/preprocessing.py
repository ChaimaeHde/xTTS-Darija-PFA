"""
preprocessing.py
Prétraitement du dataset DODa M1 pour le fine-tuning XTTS-v2.
Entrée  : dataset Kaggle (train.csv + wavs/wavs/)
Sortie  : /kaggle/working/doda_M1_3h/train.csv + wavs/
"""

import os
import shutil
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ── Chemins ──────────────────────────────────────────────────────────────────
DATASET_PATH    = "/kaggle/input/datasets/loubna11/dataset-train-xtts"
META_FILE       = os.path.join(DATASET_PATH, "train.csv")
WAV_DIR         = os.path.join(DATASET_PATH, "wavs/wavs")
LOCAL_DATA_PATH = "/kaggle/working/doda_M1_3h/"
LOCAL_WAV_DIR   = os.path.join(LOCAL_DATA_PATH, "wavs")
TRAIN_CSV       = os.path.join(LOCAL_DATA_PATH, "train.csv")

os.makedirs(LOCAL_WAV_DIR, exist_ok=True)


def copy_wavs():
    """Copie les WAVs depuis le dataset Kaggle vers /kaggle/working/."""
    existing = set(os.listdir(LOCAL_WAV_DIR))
    all_src  = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav")]
    to_copy  = [f for f in all_src if f not in existing]
    if to_copy:
        print(f"📥 Copie de {len(to_copy)} WAVs...")
        for fname in tqdm(to_copy):
            shutil.copy(os.path.join(WAV_DIR, fname),
                        os.path.join(LOCAL_WAV_DIR, fname))
        print("✅ Copie terminée")
    else:
        print(f"⏭️  {len(existing)} WAVs déjà présents")


def build_train_csv():
    """Lit train.csv, nettoie et filtre par durée (1s–11.6s)."""
    df = pd.read_csv(META_FILE, sep="|", header=None,
                     names=["audio_name", "text", "text2"],
                     on_bad_lines="skip")
    df["audio_name"] = (df["audio_name"].astype(str).str.strip()
                        .str.replace(r"\.wav$", "", regex=True))
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 3]

    # Vérifier existence WAVs
    df = df[df["audio_name"].apply(
        lambda x: os.path.exists(os.path.join(LOCAL_WAV_DIR, x + ".wav")))]

    # Filtrer durées
    valid = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtrage durées"):
        try:
            info = sf.info(os.path.join(LOCAL_WAV_DIR, row["audio_name"] + ".wav"))
            if 1.0 <= info.duration <= 11.6:
                valid.append(row["audio_name"])
        except Exception:
            pass

    df = df[df["audio_name"].isin(valid)]

    # Écriture format LJSpeech
    with open(TRAIN_CSV, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"{row['audio_name']}|{row['text']}|{row['text']}\n")

    print(f"✅ train.csv : {len(df)} samples → {TRAIN_CSV}")
    return df


if __name__ == "__main__":
    copy_wavs()
    build_train_csv()
