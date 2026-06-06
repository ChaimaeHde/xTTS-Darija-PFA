"""
finetuning.py
Fine-tuning XTTS-v2 sur le dataset Darija M1 (DODa 3h).
Plateforme : Kaggle T4x2 (GPU unique forcé via patch trainer_utils)
"""

import os
import torch
import gc
import glob
import zipfile
import pathlib
import threading
import time
import shutil

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig

# ── Chemins ──────────────────────────────────────────────────────────────────
LOCAL_DATA_PATH = "/kaggle/working/doda_M1_3h/"
MODEL_DIR       = "/kaggle/working/xtts_v2_base/"
OUT_PATH        = "/kaggle/working/xtts_M1_outputs/"

TOKENIZER_FILE  = os.path.join(MODEL_DIR, "vocab.json")
XTTS_CHECKPOINT = os.path.join(MODEL_DIR, "model.pth")
DVAE_CHECKPOINT = os.path.join(MODEL_DIR, "dvae.pth")
MEL_NORM_FILE   = os.path.join(MODEL_DIR, "mel_stats.pth")

os.makedirs(OUT_PATH, exist_ok=True)


def patch_trainer():
    """Patch trainer_utils et io.py pour Kaggle T4x2."""
    # Patch 1 : forcer 1 GPU
    p1 = "/usr/local/lib/python3.12/dist-packages/trainer/trainer_utils.py"
    with open(p1) as f: c = f.read()
    old1 = '        msg = f" [!] {num_gpus} active GPUs. Define the target GPU by `CUDA_VISIBLE_DEVICES`. For multi-gpu training use `python -m trainer.distribute`.\n        raise RuntimeError(msg)'
    new1 = '        num_gpus = 1\n        torch.cuda.set_device(0)'
    if old1 in c:
        with open(p1, "w") as f: f.write(c.replace(old1, new1))
        print("✅ trainer_utils.py patché")

    # Patch 2 : supprimer ancien best_model AVANT sauvegarder le nouveau
    p2 = "/usr/local/lib/python3.12/dist-packages/trainer/io.py"
    with open(p2) as f: c = f.read()
    old2 = '        logger.info(" > BEST MODEL : %s", checkpoint_path)\n        save_model('
    new2 = '        logger.info(" > BEST MODEL : %s", checkpoint_path)\n        import glob as _g\n        for _f in _g.glob(os.path.join(out_path, "best_model*.pth")):\n            try: os.remove(_f)\n            except: pass\n        save_model('
    if old2 in c:
        with open(p2, "w") as f: f.write(c.replace(old2, new2))
        print("✅ io.py patché")


def free_gb():
    st = os.statvfs("/kaggle/working")
    return st.f_bavail * st.f_frsize / 1e9


def watcher(pattern, done_flag, interval=15):
    print("👁️  Watcher démarré")
    while not os.path.exists(done_flag):
        for d in sorted(glob.glob(pattern)):
            for p in [os.path.join(d, "best_model.pth")]:
                if os.path.exists(p) or os.path.islink(p):
                    try: os.remove(p)
                    except: pass
            all_best = sorted(glob.glob(os.path.join(d, "best_model_*.pth")))
            for old in all_best[:-1]:
                try: os.remove(old); print(f"🗑️  {os.path.basename(old)} | {free_gb():.1f}GB")
                except: pass
        time.sleep(interval)
    print("🏁 Watcher arrêté")


def train():
    patch_trainer()
    gc.collect()
    torch.cuda.empty_cache()

    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        debug_loading_failures=False, max_wav_length=255995, max_text_length=200,
        mel_norm_file=MEL_NORM_FILE, dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT, tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    dataset_config = BaseDatasetConfig(
        dataset_name="doda_M1", path=LOCAL_DATA_PATH,
        meta_file_train="train.csv", meta_file_val="",
        ignored_speakers=None, formatter="ljspeech", language="ar",
    )

    # ── Hyperparamètres ──────────────────────────────────────────────────────
    config = GPTTrainerConfig(
        output_path=OUT_PATH, model_args=model_args,
        run_name="xtts_darija_M1_3h",
        project_name="XTTS_M1_3h_finetuning",
        run_description="XTTS-v2 fine-tuning M1 DODa 3h Darija",
        dashboard_logger="tensorboard", logger_uri=None,
        audio=audio_config,
        epochs=10,
        batch_size=2,           # contrainte VRAM 16GB
        eval_batch_size=2,
        batch_group_size=48,
        mixed_precision=False,  # évite NaN fp16
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000],
                             "gamma": 0.5, "last_epoch": -1},
        num_loader_workers=0,   # évite crashs RAM Kaggle
        eval_split_max_size=256,
        save_step=50000,
        save_n_checkpoints=1,
        save_checkpoints=False,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        print_eval=False,
        test_sentences=[],
        datasets=[dataset_config],
    )

    model = GPTTrainer.init_from_config(config)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Modèle chargé — VRAM : {torch.cuda.memory_allocated()/1e9:.1f} GB")

    train_samples, eval_samples = load_tts_samples(
        [dataset_config], eval_split=True,
        eval_split_max_size=256, eval_split_size=0.1)
    print(f"✅ Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    DONE_FLAG = "/kaggle/working/training_done.flag"
    if os.path.exists(DONE_FLAG): os.remove(DONE_FLAG)

    threading.Thread(
        target=watcher,
        args=(OUT_PATH + "run/training/*", DONE_FLAG, 15),
        daemon=True).start()

    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False,
                    start_with_eval=True, grad_accum_steps=126,
                    use_accelerate=False),
        config,
        output_path=OUT_PATH + "run/training/",
        model=model, train_samples=train_samples, eval_samples=eval_samples,
    )

    print(f"\n🚀 FINE-TUNING — 10 epochs | batch_size=2 | lr=5e-6 | grad_accum=126")
    trainer.fit()
    pathlib.Path(DONE_FLAG).touch()

    # Zip final
    TDIR = sorted(glob.glob(OUT_PATH + "run/training/*"))[-1]
    best = sorted(glob.glob(os.path.join(TDIR, "best_model_*.pth")))
    if best:
        ZIP = "/kaggle/working/xtts_M1_best_model.zip"
        with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(best[-1], "model.pth")
            zf.write(os.path.join(TDIR, "config.json"), "config.json")
            zf.write(TOKENIZER_FILE, "vocab.json")
        print(f"✅ Zip : {ZIP} ({os.path.getsize(ZIP)/1e6:.0f} MB)")


if __name__ == "__main__":
    train()
