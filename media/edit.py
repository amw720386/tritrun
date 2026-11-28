"""Cut the launch video from its two renders and the music bed, on Modal.

    modal run media/edit.py --bonsai <dir>/film.mp4 --mid <dir>/film.mp4 --music bed.wav --out <dir>

open  media/film_bonsai.py  0.0 to 12.3  the title over one bonsai; it clones into two that slide apart, PrismML's fork
                                         on the left and tritrun on the right, and both decode for 5 s, a blossom per token
mid   media/film.py         0.6 to 10.5  the block read once for four tokens, and the 64 layers
end   media/film_bonsai.py 12.3 to 17.0  back on the two crowns: tritrun's extra blossoms, the ratio and the repository

--bonsai is one render of film_bonsai.py from 0 to 17; --mid of film.py from 0 to 10.6. Each part joins the next with a
0.6 s dissolve, the first through black. The music is set to a constant gain (two-pass loudness, linear), so it never
pumps, and fades out with the picture. The bed is ACE-Step (Apache-2.0), seed 101 of the prompt in media/music.py.
"""
import io
import json
import os
import tarfile

import modal

app = modal.App("launch-edit", image=modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg"))
X = 0.6
# (part, file, span in that file's own time)
PARTS = [("open", "bonsai", 0.0, 12.3), ("mid", "mid", 0.6, 10.5), ("end", "bonsai", 12.3, None)]
CLIPS = {"against": ("open", 4.0, 11.7), "inside": ("mid", 3.2, 6.6), "layers": ("mid", 6.4, 10.5)}
JOINS = ["fadeblack", "fade"]    # the race's crowns to the block dip through black
POSTER = ("end", 15.5)           # in film_bonsai time: the crowns, the extra blossoms in blue, "1.34× faster." and the URL
MUSIC0 = 3.0                     # the bed's swell (its 23 s on) then lands on the crowns


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
    for name, f, s, e in PARTS:
        e = dur(f"/w/{f}.mp4") if e is None else e
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
            vf.append(f"[{prev}][p{i}]xfade=transition={JOINS[i - 1]}:duration={X}:offset={off:.3f}[x{i}]")
            prev, off = f"x{i}", off + e - s
    vf.append(f"[{prev}]fade=t=out:st={total - 0.8:.3f}:d=0.8[v]")
    inputs = [a for _, f, _, _ in PARTS for a in ("-i", f"/w/{f}.mp4")]
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
def main(bonsai: str = "", mid: str = "", music: str = "", out: str = "."):
    files = {k: open(p, "rb").read() for k, p in (("bonsai.mp4", bonsai), ("mid.mp4", mid), ("music.wav", music))}
    b = cut.remote(files)
    os.makedirs(out, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(b)) as tf:
        tf.extractall(out)
        print(tf.getnames())
