import os, torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pathlib import Path
import gradio as gr

os.environ["COQUI_TOS_AGREED"] = "1"

# ── Téléchargement automatique depuis HuggingFace ────────────────────────────
HF_BASE  = "https://huggingface.co/chaimaehde/xTTS_Darija_3h/resolve/main"
base_path = Path("/content/model")
base_path.mkdir(exist_ok=True)

for name, url in [
    ("config.json", f"{HF_BASE}/config.json"),
    ("vocab.json",  f"{HF_BASE}/vocab.json"),
    ("model.pth",   f"{HF_BASE}/model.pth"),
]:
    dest = base_path / name
    if not dest.exists():
        print(f"⬇️  {name}...")
        torch.hub.download_url_to_file(url, str(dest))
    else:
        print(f"⏭️  {name} déjà présent")

# ── Chargement ────────────────────────────────────────────────────────────────
print("⏳ Chargement du modèle...")
device = "cuda" if torch.cuda.is_available() else "cpu"

config = XttsConfig()
config.load_json(str(base_path / "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(config,
    checkpoint_path = str(base_path / "model.pth"),
    use_deepspeed   = False,
    vocab_path      = str(base_path / "vocab.json"),
    eval            = True,
)
model.to(device)
print(f"✅ Modèle chargé sur {device}")

# ── Interface ─────────────────────────────────────────────────────────────────
def infer(text, speaker_audio, temperature=0.75):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[speaker_audio])
    out = model.inference(text, "ar",
        gpt_cond_latent, speaker_embedding,
        temperature=temperature)
    return 24000, out["wav"]

with gr.Blocks(title="XTTS-v2 Darija M1") as demo:
    gr.Markdown("# 🎙️ XTTS-v2 Fine-tuné — Darija Marocain (M1, 3h DODa)")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Texte en Darija", lines=4, rtl=True,
                value="مرحبا كيف داير واش كلشي مزيان")
            ref_audio  = gr.Audio(label="Audio de référence", type="filepath")
            temp       = gr.Slider(0.1, 1.0, value=0.75, step=0.05,
                                   label="Temperature")
            btn        = gr.Button("🔊 Générer", variant="primary")
        out = gr.Audio(label="Audio généré")
    btn.click(infer, inputs=[text_input, ref_audio, temp], outputs=out)

if __name__ == "__main__":
    demo.launch(share=True)
