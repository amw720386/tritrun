"""Cut the launch video from its four renders and the music bed, on Modal.

    modal run media/edit.py --race <dir>/film.mp4 --opening <dir>/film.mp4 --mid <dir>/film.mp4 --end <dir>/film.mp4 --music bed.wav --out <dir>

race  media/film.py        10.6 to 16.3  both engines decode the same text on one RTX 5090, then "1.34× faster"
open  media/film_bonsai.py  0.0 to 6.4   the title over the bonsai, and the flight into one twig
mid   media/film.py         0.6 to 10.5  the block read once for four tokens, and the 64 layers
end   media/film_bonsai.py 20.4 to 26.5  the two crowns, the ratio and the repository

The race comes first so the first five seconds show the result to anyone who scrolls past. Each part joins the next
with a 0.6 s dissolve. The music is set to a constant gain (two-pass loudness, linear), so it never pumps, and fades
out with the picture. The bed is ACE-Step (Apache-2.0), seed 101 of the prompt in media/music.py.
"""
import io
import json
import os
import tarfile

import modal

app = modal.App("launch-edit", image=modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg"))
X = 0.6
# each part's span in its own file's time (race.mp4 starts at the film's 10.6, end.mp4 at 20.4)
PARTS = [("race", 0.0, None), ("open", 0.0, None), ("mid", 0.6, 10.5), ("end", 0.0, None)]
CLIPS = {"against": ("race", 0.0, 5.7), "inside": ("mid", 3.2, 6.6), "layers": ("mid", 6.4, 10.5)}
POSTER = ("end", 4.2)            # the crowns with "1.34× faster."
MUSIC0 = 3.5                     # the bed's swell (its 23 s on) then lands on the crowns


@app.function(cpu=8, memory=8192, timeout=1200)
def cut(files):
    import subprocess

    def ff(*a):
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
    spans, start, at = [], {}, 0.0
    for name, s, e in PARTS:
        e = dur(f"/w/{name}.mp4") if e is None else e
        spans.append((name, s, e))
        start[name] = at - s                      # global time = start[name] + the part's own time
        at += e - s - X
    total = at + X
    enc = ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709"]
    vf, prev, off = [], None, 0.0
    for i, (name, s, e) in enumerate(spans):
        vf.append(f"[{i}:v]trim=start={s}:end={e},settb=AVTB,setpts=PTS-STARTPTS,fps=30[p{i}]")
        if prev is None:
            prev, off = f"p{i}", e - s
        else:
            off -= X
            vf.append(f"[{prev}][p{i}]xfade=transition=fade:duration={X}:offset={off:.3f}[x{i}]")
            prev, off = f"x{i}", off + e - s
    vf.append(f"[{prev}]fade=t=out:st={total - 0.8:.3f}:d=0.8[v]")
    inputs = [a for name, _, _ in spans for a in ("-i", f"/w/{name}.mp4")]
    ff(*inputs, "-filter_complex", ";".join(vf), "-map", "[v]", *enc, "/w/picture.mp4")
    # music: trim, fade, then a constant gain to -16 LUFS
    pre = f"atrim={MUSIC0}:{MUSIC0 + total:.3f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.05,afade=t=out:st={total - 2.2:.3f}:d=2.2"
    log = ff("-i", "/w/music.wav", "-af", pre + ",loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-")
    j = json.loads(log[log.rindex("{"):log.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}")
    ff("-i", "/w/music.wav", "-af", pre + "," + ln + ",aresample=48000", "-c:a", "pcm_s16le", "/w/mix.wav")
    ff("-i", "/w/picture.mp4", "-i", "/w/mix.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
       "-shortest", "-movflags", "+faststart", "/w/launch.mp4")
    ff("-ss", f"{start[POSTER[0]] + POSTER[1]:.3f}", "-i", "/w/launch.mp4", "-frames:v", "1", "-q:v", "2", "/w/launch-poster.jpg")
    # the thread's clips, with the music under the same moments
    for name, (part, s, t) in CLIPS.items():
        g = start[part] + s
        ff("-ss", f"{g:.3f}", "-t", f"{t - s:.3f}", "-i", "/w/launch.mp4", "-af", f"afade=t=in:d=0.3,afade=t=out:st={t - s - 0.6:.3f}:d=0.6",
           "-c:v", "libx264", "-crf", "19", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"/w/{name}.mp4")
        ff("-ss", f"{g:.3f}", "-t", f"{t - s:.3f}", "-i", "/w/launch.mp4", "-vf",
           "fps=12,scale=720:-1:flags=lanczos,split[x][y];[x]palettegen=max_colors=80:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
           "-loop", "0", f"/w/{name}.gif")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for f in ["launch.mp4", "launch-poster.jpg", "mix.wav"] + [f"{n}.{x}" for n in CLIPS for x in ("mp4", "gif")]:
            tf.add(f"/w/{f}", f)
    print(f"total {total:.2f} s; parts start at " + ", ".join(f"{n} {start[n] + s:.2f}" for n, s, _ in spans) + f"; music {j['input_i']} LUFS in")
    return buf.getvalue()


@app.local_entrypoint()
def main(race: str = "", opening: str = "", mid: str = "", end: str = "", music: str = "", out: str = "."):
    files = {k: open(p, "rb").read() for k, p in (("race.mp4", race), ("open.mp4", opening), ("mid.mp4", mid), ("end.mp4", end), ("music.wav", music))}
    b = cut.remote(files)
    os.makedirs(out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b)) as tf:
        tf.extractall(out)
        print(tf.getnames())
