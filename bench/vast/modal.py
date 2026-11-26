"""Stand-in for the `modal` package, so the bench scripts can run directly on a rented machine (vast.ai).

Every attribute and call returns a stub; a decorator applied to a function returns the function unchanged.
The machine must provide what the Modal image would: /w (weights), /src (engine source), /opt/prism (fork binaries).
"""


class _Stub:
    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return _Stub()

    def __getattr__(self, name):
        return _Stub()


def __getattr__(name):
    return _Stub()
