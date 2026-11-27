"""Instrumental beds for the launch explainer with ACE-Step (Apache-2.0), on Modal.

    modal run media/music.py --out <dir> --seconds 34 --seeds 101
"""
import io
import os
import tarfile

import modal

image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "ffmpeg", "libsndfile1")
         .pip_install("git+https://github.com/ace-step/ACE-Step.git", "soundfile"))
vol = modal.Volume.from_name("ace-step-ckpt", create_if_missing=True)
app = modal.App("explainer-music", image=image)

PROMPT = ("cinematic electronic, instrumental, warm wide evolving synth pads, flowing continuous sixteenth-note arpeggio, "
          "deep sustained sub bass, soft steady pulse, smooth seamless build to an uplifting swell, modern tech product film, "
          "lush, legato, 110 bpm, no vocals, no breaks, no stops")


@app.function(gpu=["L40S", "A10", "L4"], timeout=1800, volumes={"/ckpt": vol}, memory=32768)
def gen(seconds: float, seeds: list):
    import soundfile
    import torchaudio

    def save(path, wav, sample_rate=None, *a, **k):     # torchaudio's own save now needs torchcodec
        soundfile.write(str(path), wav.detach().float().cpu().numpy().T, sample_rate or (a[0] if a else 48000))
    torchaudio.save = save
    from acestep.pipeline_ace_step import ACEStepPipeline
    pipe = ACEStepPipeline(checkpoint_dir="/ckpt", dtype="bfloat16", torch_compile=False)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for s in seeds:
            path = f"/tmp/bed_{s}.wav"
            pipe(format="wav", audio_duration=seconds, prompt=PROMPT, lyrics="[instrumental]", infer_step=100, guidance_scale=15,
                 scheduler_type="euler", cfg_type="apg", omega_scale=10, manual_seeds=[s], guidance_interval=0.5,
                 guidance_interval_decay=0.0, min_guidance_scale=3, use_erg_tag=True, use_erg_lyric=False, use_erg_diffusion=True,
                 oss_steps=[], guidance_scale_text=0.0, guidance_scale_lyric=0.0, save_path=path)
            tf.add(path, f"bed_{s}.wav")
    vol.commit()
    return buf.getvalue()


@app.local_entrypoint()
def main(out: str, seconds: float = 34.0, seeds: str = "5,19,42,77,101,133"):
    b = gen.remote(seconds, [int(x) for x in seeds.split(",")])
    os.makedirs(out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b)) as tf:
        tf.extractall(out)
        print(tf.getnames())
