"""Download the pinned Qwen3-1.7B files through a mirror; verify LFS SHA256.

The model revision and three LFS digests were read from the model repository
metadata. The mirror is only a transport, not a source of model identity.
"""
import hashlib
import json
import time
from pathlib import Path

import requests

REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FILES = {
    ".gitattributes": (1570, None),
    "LICENSE": (11343, None),
    "README.md": (13963, None),
    "config.json": (726, None),
    "generation_config.json": (239, None),
    "merges.txt": (1671853, None),
    "model-00001-of-00002.safetensors": (3441185608, "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"),
    "model-00002-of-00002.safetensors": (622329984, "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"),
    "model.safetensors.index.json": (25605, None),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9732, None),
    "vocab.json": (2776833, None),
}
OUT = Path("/data0/users/lixiaoyang/models/Qwen3-1.7B")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, expected_size, expected_sha):
    target = OUT / name
    partial = OUT / (name + ".part")
    if target.exists() and target.stat().st_size == expected_size and (not expected_sha or digest(target) == expected_sha):
        return digest(target)
    url = f"https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/{REVISION}/{name}"
    for attempt in range(60):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    partial.unlink()
                    raise RuntimeError("Server ignored range request; restarting file")
                with partial.open("ab" if offset else "wb") as out:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            out.write(chunk)
            if partial.stat().st_size != expected_size:
                raise RuntimeError(f"partial bytes {partial.stat().st_size}/{expected_size}")
            actual_sha = digest(partial)
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(f"SHA256 mismatch for {name}: {actual_sha}")
            partial.replace(target)
            print(name, expected_size, actual_sha, flush=True)
            return actual_sha
        except Exception as exc:
            print(name, "retry", attempt, type(exc).__name__, str(exc)[:100], flush=True)
            time.sleep(min(10, 1 + attempt))
    raise RuntimeError(f"Failed download: {name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    actual = {}
    for name, (size, sha) in FILES.items():
        actual[name] = {"bytes": size, "sha256": download(name, size, sha), "upstream_lfs_sha256": sha}
    (OUT / "MODEL_LOCK.json").write_text(json.dumps({"repository": "Qwen/Qwen3-1.7B",
                                                    "revision": REVISION, "files": actual}, indent=2) + "\n")
    print("MODEL_DOWNLOAD_VERIFIED", REVISION, flush=True)


if __name__ == "__main__":
    main()
