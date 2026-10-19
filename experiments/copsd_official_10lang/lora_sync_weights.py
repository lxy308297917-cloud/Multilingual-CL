"""Non-mutating equivalent of the released PEFT merged-weight export.

Scope: ordinary linear LoRA, one active adapter, no DoRA/bias/quantization.
Only the inference copy receives merged weights. Frozen teacher weights never
undergo a lossy BF16 merge/unmerge roundtrip. This is a project adaptation;
the fixed official reference implementation is retained separately.
"""
import torch
from peft.tuners.lora.layer import Linear, LoraLayer


@torch.no_grad()
def merged_lora_weight(layer):
    if not isinstance(layer, Linear) or len(layer.active_adapters) != 1:
        raise ValueError("Only one active ordinary linear LoRA adapter is supported")
    adapter = layer.active_adapters[0]
    if layer.merged or layer.disable_adapters or adapter not in layer.lora_A:
        raise ValueError("Expected an active, unmerged LoRA adapter")
    if layer.lora_variant or layer.lora_bias[adapter]:
        raise ValueError("DoRA/other LoRA variants and LoRA bias are unsupported")
    weight = layer.get_base_layer().weight
    if weight.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Quantized weights are unsupported")
    # Match PEFT merge(safe_merge=False): retain FP32 delta through add_ and
    # round once to the original base dtype. Do not cast delta before adding.
    result = weight.detach().clone()
    result.add_(layer.get_delta_weight(adapter))
    return result


def iter_merged_weights(model):
    """Yield the same exported parameter names as the official sync method."""
    if getattr(model, "modules_to_save", None):
        raise ValueError("modules_to_save is outside the validated configuration")
    layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, LoraLayer):
            if not isinstance(layer, Linear):
                raise ValueError("Only ordinary linear LoRA has been validated")
            layers[name + ".base_layer.weight"] = layer
    if not layers:
        raise ValueError("Expected a PEFT model with linear LoRA modules")
    for name, param in model.named_parameters():
        exported_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        if model.prefix in exported_name or "original_module" in exported_name:
            continue
        exported_name = exported_name.replace("modules_to_save.default.", "")
        if name in layers:
            yield exported_name, merged_lora_weight(layers[name])
        else:
            yield exported_name, param.detach()
