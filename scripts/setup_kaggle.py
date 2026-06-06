"""
scripts/setup_kaggle.py
Installation et patches nécessaires pour faire tourner
le fine-tuning XTTS-v2 sur Kaggle T4x2.
À exécuter en premier dans le notebook Kaggle.
"""

import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def install_dependencies():
    print("📦 Installation coqui-tts...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "coqui-tts", "soundfile", "librosa", "pandas",
                    "numpy", "tqdm", "huggingface_hub", "gradio"],
                   check=True)
    subprocess.run(["apt-get", "install", "-y", "-q", "ffmpeg"], check=True)
    print("✅ Dépendances installées")


def patch_trainer_utils():
    """Force l'utilisation d'un seul GPU sur Kaggle T4x2."""
    path = "/usr/local/lib/python3.12/dist-packages/trainer/trainer_utils.py"
    with open(path) as f:
        c = f.read()
    old = ('        msg = f" [!] {num_gpus} active GPUs. Define the target GPU '
           'by `CUDA_VISIBLE_DEVICES`. For multi-gpu training use '
           '`python -m trainer.distribute`.\n        raise RuntimeError(msg)')
    new = '        num_gpus = 1\n        torch.cuda.set_device(0)\n        print("[PATCH] 1 GPU forcé")'
    if old in c:
        with open(path, "w") as f:
            f.write(c.replace(old, new))
        print("✅ trainer_utils.py patché")
    elif "[PATCH]" in c:
        print("✅ trainer_utils.py déjà patché")
    else:
        print("⚠️  Pattern non trouvé dans trainer_utils.py")


def patch_io():
    """Supprime l'ancien best_model AVANT d'écrire le nouveau (évite saturation disque)."""
    path = "/usr/local/lib/python3.12/dist-packages/trainer/io.py"
    with open(path) as f:
        c = f.read()
    old = '        logger.info(" > BEST MODEL : %s", checkpoint_path)\n        save_model('
    new = ('        logger.info(" > BEST MODEL : %s", checkpoint_path)\n'
           '        import glob as _g\n'
           '        for _f in _g.glob(os.path.join(out_path, "best_model*.pth")):\n'
           '            try: os.remove(_f)\n'
           '            except: pass\n'
           '        save_model(')
    if old in c:
        with open(path, "w") as f:
            f.write(c.replace(old, new))
        print("✅ io.py patché")
    elif "PATCH: free space" in c:
        print("✅ io.py déjà patché")
    else:
        print("⚠️  Pattern non trouvé dans io.py")


def check_gpu():
    import torch
    print(f"\n🔍 PyTorch : {torch.__version__}")
    print(f"✅ CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cap   = torch.cuda.get_device_capability(0)
        print(f"✅ GPU     : {props.name}")
        print(f"✅ VRAM    : {props.total_memory/1e9:.1f} GB")
        print(f"✅ sm_{cap[0]}{cap[1]}")
        print(f"✅ GPUs visibles : {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 1:
            print("⚠️  2 GPUs détectés — patch trainer_utils appliqué")


if __name__ == "__main__":
    install_dependencies()
    patch_trainer_utils()
    patch_io()
    check_gpu()
    print("\n✅ Setup Kaggle terminé — prêt pour le fine-tuning !")
