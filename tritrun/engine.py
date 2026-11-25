"""ctypes wrapper around libtritrun.so plus a tokenizer built from the GGUF vocabulary."""
import ctypes
import os
from typing import List, Optional

from .gguf_meta import read_meta


class Tokenizer:
    """Byte-level BPE (Qwen style) built from the vocabulary embedded in the GGUF file."""

    # Qwen pre-tokenisation regex (contractions, letters, single digits, punctuation, whitespace runs)
    PATTERN = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    def __init__(self, gguf_path: str):
        from tokenizers import AddedToken, Regex, Tokenizer as HFTokenizer
        from tokenizers import decoders, models, normalizers, pre_tokenizers

        meta = read_meta(gguf_path)
        tokens = meta["tokenizer.ggml.tokens"]
        merges = meta["tokenizer.ggml.merges"]
        types = meta.get("tokenizer.ggml.token_type", [1] * len(tokens))
        self.eos_id = int(meta.get("tokenizer.ggml.eos_token_id", -1))
        vocab = {t: i for i, t in enumerate(tokens)}
        pairs = [tuple(m.split(" ", 1)) for m in merges]
        t = HFTokenizer(models.BPE(vocab=vocab, merges=pairs, byte_fallback=False))
        t.normalizer = normalizers.NFC()
        t.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(self.PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        t.decoder = decoders.ByteLevel()
        # control / user-defined tokens are matched literally
        special = [AddedToken(tok, special=True, normalized=False)
                   for tok, ty in zip(tokens, types) if ty in (3, 4)]
        t.add_special_tokens(special)
        self._t = t

    def encode(self, text: str) -> List[int]:
        return self._t.encode(text, add_special_tokens=False).ids

    def decode(self, ids: List[int]) -> str:
        return self._t.decode(ids, skip_special_tokens=True)

    def chat(self, user: str, system: Optional[str] = None) -> List[int]:
        s = ""
        if system:
            s += f"<|im_start|>system\n{system}<|im_end|>\n"
        s += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return self.encode(s)


class Engine:
    """Loads a PTQ1_0 GGUF into GPU memory and decodes greedily, with optional MTP speculation."""

    def __init__(self, gguf_path: str, ctx: int = 4096, mtp: bool = True, lib: Optional[str] = None):
        if not 16 <= ctx <= 8192:
            raise ValueError("ctx must be between 16 and 8192")
        lib = lib or os.environ.get("TRITRUN_LIB") or os.path.join(os.path.dirname(__file__), "libtritrun.so")
        self.lib = ctypes.CDLL(lib)
        L = self.lib
        L.tr_load.restype = ctypes.c_void_p
        L.tr_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.tr_generate.restype = ctypes.c_int
        L.tr_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
        L.tr_device_bytes.restype = ctypes.c_longlong
        L.tr_device_bytes.argtypes = [ctypes.c_void_p]
        self.ctx = ctx
        self.h = L.tr_load(gguf_path.encode(), ctx, 1 if mtp else 0)
        self.tok = Tokenizer(gguf_path)

    def device_bytes(self) -> int:
        return int(self.lib.tr_device_bytes(self.h))

    def generate(self, prompt_ids: List[int], max_new: int = 256, drafts: int = 3, eos: Optional[List[int]] = None):
        """Greedy decode.  Returns (token_ids, stats dict)."""
        eos = [self.tok.eos_id] if eos is None else eos
        if len(prompt_ids) + max_new > self.ctx:
            raise ValueError("prompt + max_new exceeds the context")
        arr = (ctypes.c_int * len(prompt_ids))(*prompt_ids)
        out = (ctypes.c_int * (max_new + 8))()
        e = (ctypes.c_int * max(1, len(eos)))(*eos)
        st = (ctypes.c_double * 5)()
        n = self.lib.tr_generate(self.h, arr, len(prompt_ids), max_new, e, len(eos), drafts, out, st)
        ids = [out[i] for i in range(n)]
        return ids, {
            "rounds": st[0], "accepted_drafts": st[1], "drafted": st[2], "decode_seconds": st[4],
            "decode_tok_s": (n - 1) / st[4] if st[4] > 0 and n > 1 else 0.0,
        }

    def chat(self, user: str, max_new: int = 512, drafts: int = 3, system: Optional[str] = None) -> str:
        out, _ = self.generate(self.tok.chat(user, system), max_new=max_new, drafts=drafts)
        return self.tok.decode(out)
