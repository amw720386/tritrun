"""Run one of the Modal bench scripts directly on this machine (after bench/vast/setup.sh).

python bench/vast/run.py consistency|spec|validate_long|paired|prefill|quality:lm|quality:mmlu|quality:gsm8k
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                          # the modal stub comes first
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

which = sys.argv[1]
if which == "consistency":
    import consistency; consistency.run()
elif which == "spec":
    import spec_test; spec_test.body()
elif which == "validate_long":
    import validate_long; validate_long.run()
elif which == "paired":
    import paired; paired.run_all()
elif which == "prefill":
    import prefill; prefill.run()
elif which.startswith("quality:"):
    import quality; quality.go(which.split(":")[1], 0)
else:
    raise SystemExit(f"unknown: {which}")
