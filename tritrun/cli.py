"""python -m tritrun.cli MODEL.gguf "PROMPT" [--max-new N] [--drafts 0..3] [--ctx N] [--raw]"""
import argparse
import sys

from .engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser(prog="tritrun")
    ap.add_argument("model", help="path to a PTQ1_0 GGUF (the -mtp build enables speculation)")
    ap.add_argument("prompt")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--drafts", type=int, default=3, help="MTP drafts per round, 0 disables speculation")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--raw", action="store_true", help="do not apply the chat template")
    a = ap.parse_args()

    e = Engine(a.model, ctx=a.ctx, mtp=a.drafts > 0)
    ids = e.tok.encode(a.prompt) if a.raw else e.tok.chat(a.prompt)
    out, st = e.generate(ids, max_new=a.max_new, drafts=a.drafts)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(e.tok.decode(out))
    print(f"\n[{len(out)} tokens, {st['decode_tok_s']:.1f} tok/s decode, "
          f"{st['accepted_drafts']:.0f}/{st['drafted']:.0f} drafts accepted]", file=sys.stderr)


if __name__ == "__main__":
    main()
