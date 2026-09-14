"""Optionally inject a fixed FFN-neuron ablation mask into Transformers models.

Activated only when both LANGUAGE_NEURON_MASK_FILE and
LANGUAGE_NEURON_MASK_CONDITION are set, so ordinary Python processes are
unaffected. This is loaded through PYTHONPATH by the stage-2 controller.
"""

from __future__ import annotations

import os


MASK_FILE = os.environ.get("LANGUAGE_NEURON_MASK_FILE")
CONDITION = os.environ.get("LANGUAGE_NEURON_MASK_CONDITION")


if MASK_FILE and CONDITION:
    import torch
    from transformers import AutoModelForCausalLM

    def resolve_mask():
        obj = torch.load(MASK_FILE, map_location="cpu")
        method, separator, language = CONDITION.partition(":")
        if not separator:
            raise ValueError(f"Expected METHOD:LANG, got {CONDITION!r}")
        if method.startswith("random_"):
            source = obj["random_masks"]
            method = method.removeprefix("random_")
        else:
            source = obj["masks"]
        return source[method][language]

    def attach_mask(model):
        masks = resolve_mask()
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) != len(masks):
            raise RuntimeError("Language-neuron mask does not match model.model.layers")
        handles = []
        device_masks = {}
        for layer_id, layer in enumerate(layers):
            def hook(_module, inputs, index=layer_id):
                value = inputs[0]
                if masks[index].numel() == 0:
                    return None
                key = (index, value.device)
                chosen = device_masks.get(key)
                if chosen is None:
                    chosen = masks[index].to(value.device)
                    device_masks[key] = chosen
                value = value.clone()
                value.index_fill_(-1, chosen, 0)
                return (value,)
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        model._language_neuron_mask_handles = handles
        model._language_neuron_mask_condition = CONDITION
        return model

    original_from_pretrained = AutoModelForCausalLM.from_pretrained.__func__

    def masked_from_pretrained(cls, *args, **kwargs):
        return attach_mask(original_from_pretrained(cls, *args, **kwargs))

    AutoModelForCausalLM.from_pretrained = classmethod(masked_from_pretrained)
