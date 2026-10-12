"""No real credentials in tracked files, and secret files stay ignored."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Shapes of real keys for the supported providers (placeholders such as
# "sk-ant-your-key-here" or "AIza...your-key" don't match).
KEY_PATTERNS = [
    r"sk-ant-[A-Za-z0-9_-]{30,}", r"sk-proj-[A-Za-z0-9_-]{30,}", r"sk-or-v1-[a-f0-9]{40,}",
    r"gsk_[A-Za-z0-9]{40,}", r"hf_[A-Za-z0-9]{30,}", r"xai-[A-Za-z0-9]{40,}", r"AIza[0-9A-Za-z_-]{35}",
]


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / f for f in out.splitlines() if f]


def test_no_real_keys_in_tracked_files():
    pattern = re.compile("|".join(KEY_PATTERNS))
    hits = []
    for path in _tracked_files():
        if path.suffix in {".png", ".jpg", ".ico", ".db"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{path.relative_to(ROOT)}: {m.group(0)[:8]}…" for m in pattern.finditer(text)]
    assert not hits, f"possible credentials committed: {hits}"


def test_secret_files_are_ignored_and_untracked():
    tracked = {str(p.relative_to(ROOT)) for p in _tracked_files()}
    assert not {".env", ".streamlit/secrets.toml"} & tracked
    ignore = (ROOT / ".gitignore").read_text()
    for entry in (".env", "secrets.toml", "*.db-wal"):
        assert entry in ignore
