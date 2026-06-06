"""
scripts/download_model.py
Télécharge le modèle fine-tuné depuis HuggingFace et l'extrait dans ./model/
"""

import os
import zipfile
from huggingface_hub import hf_hub_download

REPO_ID   = "chaimaehde/xTTS_Darija_3h"
FILENAME  = "xtts_M1_best_model.zip"
MODEL_DIR = "./model"

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"⬇️  Téléchargement depuis {REPO_ID}...")
zip_path = hf_hub_download(
    repo_id   = REPO_ID,
    filename  = FILENAME,
    local_dir = MODEL_DIR,
)
print(f"✅ Téléchargé : {zip_path}")

print("📦 Extraction...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(MODEL_DIR)

print("📂 Contenu du dossier model/ :")
for f in os.listdir(MODEL_DIR):
    size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1e6
    print(f"   {f} ({size:.0f} MB)")

print("\n✅ Modèle prêt dans ./model/")
