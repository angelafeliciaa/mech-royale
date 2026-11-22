"""Load a Bot class from the working tree or from any git ref (`git show <ref>:dex_bot.py` etc.)."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOT_MODULES = ("dex_bot", "dex_combat", "dex_nav", "dex_tactics", "arena_map", "mechfight")


def export_ref(ref: str) -> Path:
    """Materialize root-level *.py of `ref` into a temp dir and return it."""
    out = Path(tempfile.mkdtemp(prefix=f"mrbot-{ref.replace('/', '_')}-"))
    names = subprocess.run(["git", "ls-tree", "--name-only", ref], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.split()
    for name in names:
        if name.endswith(".py"):
            src = subprocess.run(["git", "show", f"{ref}:{name}"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout
            (out / name).write_text(src)
    return out


def load_bot_class(spec: str):
    """spec: 'working' (repo root as checked out), a git ref, or a directory path."""
    if spec == "working":
        d = ROOT
    elif os.path.isdir(spec):
        d = Path(spec).resolve()
    else:
        d = export_ref(spec)
    saved = {k: sys.modules.pop(k) for k in BOT_MODULES if k in sys.modules}
    sys.path.insert(0, str(d))
    try:
        mod = importlib.import_module("dex_bot")
        cls = mod.Bot
    finally:
        sys.path.remove(str(d))
        for k in BOT_MODULES:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    cls._sim_source = str(d)
    return cls
