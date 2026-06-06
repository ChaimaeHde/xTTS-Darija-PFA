"""
inference.py
Inférence avec le modèle XTTS-v2 fine-tuné sur Darija M1.
"""

import os
import gc
import torch
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

os.environ["COQUI_TOS_AGREED"] = "1"

# ── Chemins ──────────────────────────────────────────────────────────────────
MODEL_DIR  = "./model"   # dossier contenant model.pth, config.json, vocab.json
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_dir: str = MODEL_DIR):
    """Charge le modèle fine-tuné."""
    gc.collect()
    torch.cuda.empty_cache()

    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=os.path.join(model_dir, "model.pth"),
        vocab_path=os.path.join(model_dir, "vocab.json"),
        eval=True,
    )
    if torch.cuda.is_available():
        model.cuda()
    print("✅ Modèle chargé !")
    return model, config


def synthesize(model, config, text: str, speaker_wav: str,
               output_path: str = None, language: str = "ar"):
    """
    Génère un fichier audio depuis un texte en Darija.

    Args:
        model       : modèle XTTS chargé
        config      : config XTTS
        text        : texte en Darija (arabe)
        speaker_wav : chemin vers un WAV de référence du locuteur
        output_path : chemin de sortie (optionnel)
        language    : code langue (défaut : "ar")

    Returns:
        numpy array du signal audio (24000 Hz)
    """
    outputs = model.synthesize(
        text=text,
        config=config,
        speaker_wav=speaker_wav,
        language=language,
    )
    wav = outputs["wav"]

    if output_path:
        sf.write(output_path, wav, 24000)
        print(f"✅ Audio sauvegardé : {output_path}")

    return wav


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inférence XTTS Darija")
    parser.add_argument("--text",        type=str, required=True,
                        help="Texte en Darija à synthétiser")
    parser.add_argument("--speaker_wav", type=str, required=True,
                        help="Chemin vers le WAV de référence")
    parser.add_argument("--output",      type=str, default="output.wav",
                        help="Fichier de sortie")
    parser.add_argument("--model_dir",   type=str, default=MODEL_DIR,
                        help="Dossier du modèle fine-tuné")
    args = parser.parse_args()

    model, config = load_model(args.model_dir)
    synthesize(model, config, args.text, args.speaker_wav, args.output)
