"""Cut the launch video from its three renders and the music bed, on Modal.

    modal run media/edit.py --opening <dir>/film.mp4 --mid <dir>/film.mp4 --end <dir>/film.mp4 --music bed.wav --out <dir>

open  media/film_bonsai.py  0.0 to 6.4   the title over the bonsai, and the flight into one twig
mid   media/film.py         0.6 to 16.2  the block, the 64 layers, and the race on two pages
end   media/film_bonsai.py 20.4 to 26.5  the two crowns, the ratio and the repository

Each joins the next with a 0.6 s dissolve. The music is set to a constant gain (two-pass loudness, linear), so it
never pumps, and fades out with the picture. The bed is ACE-Step (Apache-2.0), seed 101 of the prompt in
media/music.py.
"""
import io
import json
import os
import tarfile

import modal

app = modal.App("launch-edit", image=modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg"))
X = 0.6
MID0 = 0.6
CLIPS = {"inside": (3.2, 6.6), "layers": (6.4, 10.5), "against": (10.6, 15.6)}     # in mid's own time
POSTER = 25.0
MUSIC0 = 0.5                     # the bed's accents (every 3.3 s) then land on the cuts


@app.function(cpu=8, memory=8192, timeout=1200)
def cut(files):
    import subprocess

    def ff(*a, cap=False):
        r = subprocess.run(["ffmpeg", "-y", "-hide_banner", *a], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-1500:])
        return r.stderr

    def dur(p):
        return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                    capture_output=True, text=True, check=True).stdout)
    os.makedirs("/w", exist_ok=True)
    for k, b in files.items():
        open(f"/w/{k}", "wb").write(b)
    a, m, e = dur("/w/open.mp4"), dur("/w/mid.mp4") - MID0, dur("/w/end.mp4")
    o1 = a - X
    o2 = o1 + m - X
    total = o2 + e
    enc = ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709"]
    vf = (f"[0:v]settb=AVTB,setpts=PTS-STARTPTS,fps=30[o];[1:v]trim=start={MID0},settb=AVTB,setpts=PTS-STARTPTS,fps=30[m];"
          f"[2:v]settb=AVTB,setpts=PTS-STARTPTS,fps=30[e];"
          f"[o][m]xfade=transition=fade:duration={X}:offset={o1:.3f}[ab];"
          f"[ab][e]xfade=transition=fade:duration={X}:offset={o2:.3f},fade=t=in:st=0:d=0.6,fade=t=out:st={total - 0.8:.3f}:d=0.8[v]")
    ff("-i", "/w/open.mp4", "-i", "/w/mid.mp4", "-i", "/w/end.mp4", "-filter_complex", vf, "-map", "[v]", *enc, "/w/picture.mp4")
    # music: trim, fade, then a constant gain to -16 LUFS
    pre = f"atrim={MUSIC0}:{MUSIC0 + total:.3f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.4,afade=t=out:st={total - 2.2:.3f}:d=2.2"
    log = ff("-i", "/w/music.wav", "-af", pre + ",loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-", cap=True)
    j = json.loads(log[log.rindex("{"):log.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}")
    ff("-i", "/w/music.wav", "-af", pre + "," + ln + ",aresample=48000", "-c:a", "pcm_s16le", "/w/mix.wav")
    ff("-i", "/w/picture.mp4", "-i", "/w/mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
       "-shortest", "-movflags", "+faststart", "/w/launch.mp4")
    ff("-ss", str(POSTER), "-i", "/w/launch.mp4", "-frames:v", "1", "-q:v", "2", "/w/launch-poster.jpg")
    # the thread's clips, with the music under the same moments
    for name, (s, t) in CLIPS.items():
        g = o1 + (s - MID0)
        ff("-ss", f"{g:.3f}", "-t", f"{t - s:.3f}", "-i", "/w/launch.mp4", "-af", f"afade=t=in:d=0.3,afade=t=out:st={t - s - 0.6:.3f}:d=0.6",
           "-c:v", "libx264", "-crf", "19", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"/w/{name}.mp4")
        ff("-ss", f"{g:.3f}", "-t", f"{t - s:.3f}", "-i", "/w/launch.mp4", "-vf",
           "fps=12,scale=720:-1:flags=lanczos,split[x][y];[x]palettegen=max_colors=80:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
           "-loop", "0", f"/w/{name}.gif")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for f in ["launch.mp4", "launch-poster.jpg", "mix.wav"] + [f"{n}.{x}" for n in CLIPS for x in ("mp4", "gif")]:
            tf.add(f"/w/{f}", f)
    print(f"total {total:.2f} s, music {j['input_i']} LUFS in")
    return buf.getvalue()


@app.local_entrypoint()
def main(opening: str = "", mid: str = "", end: str = "", music: str = "", out: str = "."):
    files = {k: open(p, "rb").read() for k, p in (("open.mp4", opening), ("mid.mp4", mid), ("end.mp4", end), ("music.wav", music))}
    b = cut.remote(files)
    os.makedirs(out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b)) as tf:
        tf.extractall(out)
        print(tf.getnames())
