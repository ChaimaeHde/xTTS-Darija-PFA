#  XTTS-v2 Fine-tuning — Darija Marocain (M1, DODa 3h)

## Description
Fine-tuning du modèle **XTTS-v2** (Coqui TTS) sur ~3h d'audio en **Darija marocain**
extrait du dataset **DODa** (locuteur M1), dans le cadre d'un projet de recherche
sur la synthèse vocale pour les dialectes arabes.

## Résultats
| Métrique | Valeur |
|----------|--------|
| Epochs | 10 |
| loss_mel_ce finale | 3.41 |
| Plateforme | Kaggle T4 GPU |
| Durée training | ~2h30 |

## Structure du projet
.
├── app.py                  # Interface Gradio (lancement)
├── requirements.txt        # Dépendances
├── src/
│   ├── preprocessing.py    # Prétraitement dataset DODa
│   ├── finetuning.py       # Fine-tuning XTTS-v2
│   └── inference.py        # Inférence / synthèse vocale
└── model/                  # (non inclus) model.pth + config.json + vocab.json

## Installation
```bash
pip install -r requirements.txt
```

## Télécharger le modèle
```python
from huggingface_hub import hf_hub_download
import zipfile

path = hf_hub_download(
    repo_id  = "chaimaehde/xTTS_Darija_3h",
    filename = "xtts_M1_best_model.zip",
    local_dir= "./model"
)
with zipfile.ZipFile(path, "r") as zf:
    zf.extractall("./model")
```

## Lancer l'interface
```bash
python app.py
```

## Inférence en ligne de commande
```bash
python src/inference.py \
  --text "مرحبا، كيف داير؟" \
  --speaker_wav reference.wav \
  --output output.wav \
  --model_dir ./model
```

## Hyperparamètres du fine-tuning
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Modèle de base | XTTS-v2 | Support natif arabe |
| Dataset | 3h Darija (LJSpeech) | DODa locuteur M1 |
| Plateforme | Kaggle T4 GPU | 16GB VRAM |
| batch_size | 2 | Contrainte VRAM |
| grad_accum_steps | 126 | Batch effectif = 252 |
| learning_rate | 5e-6 | Fine-tuning conservateur |
| optimizer | AdamW (β1=0.9, β2=0.96) | Standard TTS |
| epochs | 10 | Convergence loss < 3.5 |
| mixed_precision | False | Évite NaN fp16 |
| num_workers | 0 | Stabilité Kaggle |
| sample_rate | 22 050 Hz | Requis XTTS-v2 |

## Dataset
- **DODa** (Darija Open Dataset) — locuteur M1
- ~3 983 segments audio après prétraitement
- Filtrage durées : 1s–11.6s
- Format : LJSpeech (id|texte|texte)

## Modèle sur HuggingFace
 [chaimaehde/xTTS_Darija_3h](https://huggingface.co/chaimaehde/xTTS_Darija_3h)

## Auteurs

| Nom | Formation |
|-----|-----------|
| **Haouach Loubna** | 2ème année INDIA-SD — ENSAM Rabat |
| **Haddouche Chaimae** | 2ème année INDIA-SD — ENSAM Rabat |

Projet réalisé dans le cadre du projet de fin d'année 
**École Nationale Supérieure d'Arts et Métiers — Rabat, Maroc**  
Année universitaire 2024–2025

## Crédits
- **XTTS-v2** : [Coqui TTS](https://github.com/coqui-ai/TTS) — modèle de synthèse vocale multilingue
- **DODa** : [Darija Open Dataset](https://github.com/AIOXLABS/DODa) — dataset audio Darija marocain
- **Coqui Trainer** : framework de fine-tuning utilisé pour l'entraînement
- **HuggingFace Hub** : hébergement du modèle fine-tuné
- **Kaggle** : plateforme GPU utilisée pour le fine-tuning (T4, 16GB VRAM)

