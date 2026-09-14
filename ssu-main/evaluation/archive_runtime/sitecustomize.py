"""Apply a verified Language-Shell structured-slice archive at model load."""

from __future__ import annotations

import os


ARCHIVE_PATH = os.environ.get("LANGUAGE_SHELL_ARCHIVE")


if ARCHIVE_PATH:
    import torch
    from transformers import AutoModelForCausalLM

    archive = torch.load(ARCHIVE_PATH, map_location="cpu", weights_only=False)
    if archive.get("format") != "language_shell_structured_slices_v1":
        raise RuntimeError(f"Unsupported Language-Shell archive: {ARCHIVE_PATH}")

    def apply_archive(model):
        with torch.no_grad():
            for layer_id in archive["manifest"]["layers"]:
                indices = archive["tensors"][f"layers.{layer_id}.indices"].long()
                mlp = model.model.layers[int(layer_id)].mlp
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = f"model.layers.{layer_id}.mlp.{projection}.weight"
                    target = getattr(mlp, projection).weight
                    values = archive["tensors"][key].to(device=target.device, dtype=target.dtype)
                    if projection == "down_proj":
                        target[:, indices.to(target.device)] = values
                    else:
                        target[indices.to(target.device), :] = values
        model._language_shell_archive = ARCHIVE_PATH
        return model

    original_from_pretrained = AutoModelForCausalLM.from_pretrained.__func__

    def archived_from_pretrained(cls, *args, **kwargs):
        return apply_archive(original_from_pretrained(cls, *args, **kwargs))

    AutoModelForCausalLM.from_pretrained = classmethod(archived_from_pretrained)
