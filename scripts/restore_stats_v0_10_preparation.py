"""Restore the audited v0.10 preparation and v0.9 adapter from Kaggle v15.

Run inside Kaggle after removing a conflicting same-notebook input.
No package installation, teacher requests, or training is performed.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

PREP_SHA = "f5772cb391f1bbd342a0e3f278e39a1a2bbe554ce72b215af139d9dbb0d2fb6c"
ADAPTER_SHA = "805a2170a805f6176aa3837857890b8c44fc8f854d16cbc3085ae220e5502c7c"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    import kagglehub

    saved = Path(kagglehub.notebook_output_download(
        "trinashih/3beethoven-v0-2/versions/15"
    ))
    working = Path("/kaggle/working")
    source = saved / "3beethoven_stats_v0_9"
    archive = saved / "3beethoven_stats_v0_10.zip"
    assert sha(archive) == PREP_SHA
    assert sha(source / "adapter/adapter_model.safetensors") == ADAPTER_SHA
    # Version 15 preserves the v0.9 directory, not the original v0.9 ZIP.
    manifest = json.loads((source / "manifest.json").read_text())
    for item in manifest:
        p = source / item["path"]
        assert p.resolve().is_relative_to(source.resolve())
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
    new = working / "3beethoven_stats_v0_10"
    old = working / "3beethoven_stats_v0_9"
    assert not new.exists() and not old.exists(), "Refuse to overwrite existing work"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all((new / n).resolve().is_relative_to(new.resolve()) for n in z.namelist())
        new.mkdir()
        z.extractall(new)
    shutil.copy2(archive, working / archive.name)
    shutil.copytree(source, old)
    assert sha(old / "adapter/adapter_model.safetensors") == ADAPTER_SHA
    print("Restored audited preparation and exact v0.9 weights; no training run")


if __name__ == "__main__":
    main()
