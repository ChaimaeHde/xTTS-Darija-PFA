"""
src/gradio_demo.py
Interface Gradio complète pour démonstration et évaluation MOS.
Peut être lancée standalone ou importée dans app.py.
"""

import os
import tempfile
import gradio as gr
import soundfile as sf

os.environ["COQUI_TOS_AGREED"] = "1"


def build_demo(model, config):
    """
    Construit et retourne l'interface Gradio.

    Args:
        model  : modèle XTTS chargé via inference.load_model()
        config : config XTTS

    Returns:
        gr.Blocks : interface Gradio
    """

    def generate(text, speaker_wav, language):
        if not text or not text.strip():
            return None, "⚠️ Texte vide"
        if speaker_wav is None:
            return None, "⚠️ Audio de référence manquant"
        try:
            outputs = model.synthesize(
                text=text,
                config=config,
                speaker_wav=speaker_wav,
                language=language,
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, outputs["wav"], 24000)
            duration = len(outputs["wav"]) / 24000
            return tmp.name, f"✅ Audio généré ({duration:.1f}s)"
        except Exception as e:
            return None, f"❌ Erreur : {str(e)}"

    with gr.Blocks(
        title="XTTS-v2 Darija M1",
        theme=gr.themes.Soft(),
        css=".rtl { direction: rtl; text-align: right; }"
    ) as demo:

        gr.Markdown("""
        # 🎙️ XTTS-v2 Fine-tuné — Darija Marocain
        **Modèle** : XTTS-v2 fine-tuné sur 3h de données DODa (locuteur M1)  
        **Langue** : Darija marocaine | **Sample rate** : 24 000 Hz
        """)

        with gr.Tabs():

            # ── Onglet Principal ─────────────────────────────────────────────
            with gr.Tab("🎤 Synthèse"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Texte en Darija",
                            placeholder="اكتب هنا بالدارجة المغربية...",
                            lines=4,
                            elem_classes=["rtl"],
                        )
                        language = gr.Dropdown(
                            choices=["ar"],
                            value="ar",
                            label="Langue",
                        )
                        ref_audio = gr.Audio(
                            label="Audio de référence (locuteur M1)",
                            type="filepath",
                        )
                        btn = gr.Button(
                            "🔊 Générer", variant="primary", size="lg")

                    with gr.Column():
                        audio_out = gr.Audio(
                            label="Audio généré", type="filepath")
                        status = gr.Textbox(
                            label="Statut", interactive=False)

                btn.click(
                    fn=generate,
                    inputs=[text_input, ref_audio, language],
                    outputs=[audio_out, status],
                )

                gr.Examples(
                    examples=[
                        ["مرحبا، كيف داير؟ واش كلشي مزيان معاك اليوم؟"],
                        ["الجو مزيان بزاف اليوم، خرجنا نتفرجو فالمدينة."],
                        ["واش عندك شي حاجة خاصة تدير اليوم؟"],
                        ["الله يحفظك، بارك الله فيك على كل شي."],
                        ["كنت كنتمنى نشوف الفيلم ديال الليلة معاك."],
                        ["الطقس بارد بزاف هاد الأيام، خاصنا نلبسو كتان."],
                        ["واش مشيتي للسوق اليوم؟ جبتيلي شي حاجة؟"],
                        ["الناس فالمغرب كيحبو يشربو أتاي بالنعناع."],
                        ["عندي امتحان غدا، خاصني نقرا بزاف الليلة."],
                        ["بلادنا فيها بزاف د الأماكن الجميلة كيفاش مراكش وفاس."],
                    ],
                    inputs=[text_input],
                    label="Phrases de test Darija (MOS)",
                )

            # ── Onglet Infos ─────────────────────────────────────────────────
            with gr.Tab("ℹ️ Infos modèle"):
                gr.Markdown("""
                ## Détails du modèle

                | Paramètre | Valeur |
                |-----------|--------|
                | Modèle de base | XTTS-v2 (coqui/XTTS-v2) |
                | Dataset | DODa M1 (~3h, 3983 segments) |
                | Epochs | 10 |
                | batch_size | 2 |
                | grad_accum_steps | 126 (batch effectif = 252) |
                | learning_rate | 5e-6 |
                | optimizer | AdamW (β1=0.9, β2=0.96) |
                | loss_mel_ce finale | 3.41 |
                | Plateforme | Kaggle T4 GPU |

                ## Utilisation
                1. Uploadez un audio de référence du locuteur M1 (3–6 secondes)
                2. Entrez votre texte en Darija
                3. Cliquez sur **Générer**

                ## Limitations
                - Optimisé pour la voix du locuteur M1 du dataset DODa
                - Certains phonèmes arabes spécifiques (غ، خ، ق، ع) peuvent être imprécis
                - Meilleurs résultats avec des segments courts (< 10 secondes)

                ## Liens
                - 🤗 [Modèle HuggingFace](https://huggingface.co/chaimaehde/xTTS_Darija_3h)
                - 📄 [DODa Dataset](https://github.com/AIOXLABS/DODa)
                - 🔧 [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
                """)

    return demo


if __name__ == "__main__":
    from src.inference import load_model
    model, config = load_model("./model")
    demo = build_demo(model, config)
    demo.launch(share=True)
