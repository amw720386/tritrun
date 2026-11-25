"""Read the key/value header of a GGUF v3 file (tensor data is not touched)."""
import struct


def read_meta(path: str) -> dict:
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize("<" + fmt)
            b = f.read(n)
            if len(b) != n:
                raise ValueError("truncated gguf header")
            return struct.unpack("<" + fmt, b)[0]

        def rstr():
            return f.read(rd("Q")).decode("utf-8", "replace")

        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("I"), rd("Q")
                return [rval(et) for _ in range(n)]
            return rd(scalar[t])

        if f.read(4) != b"GGUF" or rd("I") != 3:
            raise ValueError("not a GGUF v3 file")
        rd("Q")  # tensor count
        n_kv = rd("Q")
        meta = {}
        for _ in range(n_kv):
            key = rstr()
            meta[key] = rval(rd("I"))
        return meta
