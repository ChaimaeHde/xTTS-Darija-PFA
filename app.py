import subprocess, sys, os, zipfile, tempfile

os.environ["COQUI_TOS_AGREED"] = "1"

# ── Téléchargement automatique du modèle ─────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
HF_REPO   = "chaimaehde/xTTS_Darija_3h"
HF_FILE   = "xtts_M1_best_model.zip"

if not os.path.exists(os.path.join(MODEL_DIR, "model.pth")):
    from huggingface_hub import hf_hub_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("⬇️  Téléchargement du modèle (~5.1 GB)...")
    zip_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE,
                                local_dir=MODEL_DIR)
    print("📦 Extraction...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODEL_DIR)
    print("✅ Modèle prêt !")
else:
    print("✅ Modèle déjà présent")

# ── Chargement ────────────────────────────────────────────────────────────────
from src.inference import load_model, synthesize
import gradio as gr

model, config = load_model(MODEL_DIR)

# ── Interface ─────────────────────────────────────────────────────────────────
def generate_speech(text, speaker_wav):
    if not text or not text.strip() or speaker_wav is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    synthesize(model, config, text, speaker_wav, tmp.name)
    return tmp.name

with gr.Blocks(title="XTTS-v2 Darija M1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ XTTS-v2 — Darija Marocain (M1, 3h DODa)")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Texte en Darija", lines=4, rtl=True,
                placeholder="اكتب هنا بالدارجة المغربية...")
            ref_audio  = gr.Audio(label="Audio de référence", type="filepath")
            btn        = gr.Button("🔊 Générer", variant="primary")
        audio_out = gr.Audio(label="Audio généré", type="filepath")
    btn.click(fn=generate_speech, inputs=[text_input, ref_audio], outputs=audio_out)

if __name__ == "__main__":
    demo.launch(share=True)
