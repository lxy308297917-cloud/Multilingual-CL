"""Shared immutable prompt and identifier rules for math single SFT."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
LANGUAGE_NAMES = {"sw": "Swahili", "te": "Telugu", "th": "Thai", "bn": "Bengali", "zh": "Chinese", "ja": "Japanese", "ru": "Russian", "de": "German", "fr": "French", "es": "Spanish", "en": "English"}


def normalize_question(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def messages(question, lang):
    name = LANGUAGE_NAMES[lang]
    return [{"role": "system", "content": f"Solve the mathematics problem step by step in {name}. Use {name} for reasoning and final explanation. End with one \\boxed{{final answer}}."}, {"role": "user", "content": question}]


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
