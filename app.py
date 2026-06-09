"""
app.py
Interface Gradio pour le modèle XTTS-v2 fine-tuné sur Darija M1.
Usage : python app.py
"""

import os
import zipfile
import tempfile
import gradio as gr

os.environ["COQUI_TOS_AGREED"] = "1"

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
HF_REPO   = "chaimaehde/xTTS_Darija_3h"
HF_FILE   = "xtts_M1_best_model.zip"

# ── Téléchargement automatique du modèle ─────────────────────────────────────
def download_model():
    if os.path.exists(os.path.join(MODEL_DIR, "model.pth")):
        print("✅ Modèle déjà présent")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"⬇️  Téléchargement du modèle depuis HuggingFace (~5.1 GB)...")
    zip_path = hf_hub_download(
        repo_id   = HF_REPO,
        filename  = HF_FILE,
        local_dir = MODEL_DIR,
    )
    print("📦 Extraction...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODEL_DIR)
    print("✅ Modèle prêt !")

download_model()

# ── Chargement du modèle ──────────────────────────────────────────────────────
from src.inference import load_model, synthesize
model, config = load_model(MODEL_DIR)

# ── Callback Gradio ───────────────────────────────────────────────────────────
def generate_speech(text: str, speaker_wav: str):
    if not text or not text.strip():
        return None
    if speaker_wav is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    synthesize(model, config, text, speaker_wav, tmp.name)
    return tmp.name

# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="XTTS-v2 Darija M1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ XTTS-v2 Fine-tuné — Darija Marocain (M1, 3h DODa)
    Synthèse vocale en Darija (arabe marocain) avec clonage vocal.
    Uploadez un audio de référence du locuteur M1, entrez votre texte et générez !
    """)
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 Texte en Darija",
                placeholder="اكتب هنا بالدارجة المغربية...",
                lines=4,
                rtl=True,
            )
            ref_audio = gr.Audio(
                label="🎤 Audio de référence (locuteur M1)",
                type="filepath",
            )
            btn = gr.Button("🔊 Générer l'audio", variant="primary", size="lg")
        with gr.Column(scale=1):
            audio_out = gr.Audio(label="🔈 Audio généré", type="filepath")

    gr.Examples(
        examples=[
            ["مرحبا، كيف داير؟ واش كلشي مزيان معاك اليوم؟"],
            ["الجو مزيان بزاف اليوم، خرجنا نتفرجو فالمدينة."],
            ["واش عندك شي حاجة خاصة تدير اليوم؟"],
            ["الله يحفظك، بارك الله فيك على كل شي."],
            ["كنت كنتمنى نشوف الفيلم ديال الليلة معاك."],
        ],
        inputs=[text_input],
        label="Exemples de phrases Darija",
    )
    gr.Markdown("""
    ---
    **Modèle** : XTTS-v2 fine-tuné sur ~3h d'audio DODa (locuteur M1)  
    **Langue** : Darija marocaine (ar)  
    **Sample rate** : 24 000 Hz  
    **Projet** : Fine-tuning XTTS pour le Darija — Master 2
    """)
    btn.click(fn=generate_speech, inputs=[text_input, ref_audio], outputs=audio_out)

if __name__ == "__main__":
    demo.launch(share=True)
