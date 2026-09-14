"""
Model utilities for handling different architectures.
"""
import torch
import random
import os
from typing import Dict, Optional, Union
from transformers import AutoModel, AutoModelForCausalLM
import numpy as np
import gc


def _make_scale_hook(scale_mask):
    """
    SG-SSU 使用的梯度缩放 hook。

    原来的 _make_mask_hook 只能做二值冻结：
        mask=True  -> grad = 0
        mask=False -> grad 保持不变

    这里的 _make_scale_hook 可以做连续缩放：
        scale=0.0 -> 完全冻结，相当于不更新
        scale=0.1 -> 只保留 10% 梯度，小幅更新
        scale=0.5 -> 保留 50% 梯度，中等更新
        scale=1.0 -> 正常更新

    这样就可以实现：
        Old-Specific           scale=0.0
        Old-Shared             scale=0.1
        Current-History-Shared scale=0.5
        Current-Specific       scale=1.0
        Others                 scale=1.0

    注意：
        这是 gradient scaling 快速版。
        在 AdamW 下，grad scaling 不完全等价于真正的 per-parameter learning rate scaling，
        但它实现简单，适合第一版快速验证。
    """
    def hook_fn(grad):
        if grad is None:
            return grad

        # 保证 scale_mask 和 grad 在同一个 device、同一个 dtype 上，
        # 避免 CPU/CUDA/NPU device mismatch 报错。
        s = scale_mask.to(device=grad.device, dtype=grad.dtype)

        # 核心逻辑：不同位置的梯度乘以不同 scale。
        return grad * s

    return hook_fn





def _make_mask_hook(mask, invert=False):
    """
    Create a gradient hook that always moves mask to the same device as grad.
    This avoids errors like:
    expected self and mask to be on the same device, but got mask on cpu and self on cuda:0
    """
    def hook_fn(grad):
        if grad is None:
            return grad
        m = mask.to(device=grad.device, dtype=torch.bool)
        if invert:
            m = ~m
        return grad.masked_fill(m, 0.0)
    return hook_fn



def _should_skip_module(module_name, skip_embeddings_and_head=False):
    """
    Check if a module should be skipped from freezing.
    
    Args:
        module_name: Name of the module
        skip_embeddings_and_head: Whether to skip embedding layers and lm_head
    
    Returns:
        bool: True if module should be skipped, False otherwise
    """
    if not skip_embeddings_and_head:
        return False
    
    name_lower = module_name.lower()
    
    # Common embedding layer names
    embedding_keywords = ['embed', 'embedding', 'wte', 'wpe', 'embed_tokens', 'token_embedding', 'word_embedding']
    
    # Common lm_head layer names
    lm_head_keywords = ['lm_head', 'lm_head_layer', 'output_layer', 'head', 'classifier', 'output_projection']
    
    # Check if module name contains any embedding or lm_head keywords
    for keyword in embedding_keywords + lm_head_keywords:
        if keyword in name_lower:
            return True
    
    return False


def freeze_random_parameters(model, freeze_ratio=0.5, seed=None, strategy="random_based", skip_embeddings_and_head=False, calibration_data=None, num_calibration_samples=128, tokenizer=None, freeze_chat_template_tokens=False, chat_template_freeze_ratio=1.0):
    """
    Randomly freeze parameters in the model using the specified strategy.
    
    Args:
        model: The model to apply random freezing to
        freeze_ratio: Ratio of parameters to freeze (0.0 to 1.0)
        seed: Random seed for reproducibility
    strategy: "random_based" for individual random neurons and weights freezing by columns,
         "random_rowwise" for random row-wise freezing,
         "random_elementwise" for random element-wise freezing,
         "hft_based" for HFT-based freezing,
         "magnitude_based" for magnitude-based freezing by columns,
         "magnitude_rowwise" for row-wise magnitude-based freezing,
         "magnitude_elementwise" for element-wise magnitude-based freezing,
         "ssu_based" for SSU-based freezing,
         "ssu_rowwise" for row-wise SSU-based freezing,
         "ssu_elementwise" for element-wise SSU-based freezing,
         "fisher_based" for Fisher-based freezing by columns,
         "fisher_rowwise" for row-wise Fisher-based freezing,
         "fisher_elementwise" for element-wise Fisher-based freezing,
         "sgpt_based" for SparseGPT-based freezing by columns,
         "sgpt_rowwise" for row-wise SparseGPT-based freezing,
         "sgpt_elementwise" for element-wise SparseGPT-based freezing,
        skip_embeddings_and_head: Whether to skip freezing embedding layers and lm_head
        calibration_data: DataLoader or iterable with calibration samples (for ssu_based strategy)
        num_calibration_samples: Number of samples to use for calibration (for ssu_based strategy)
        tokenizer: Tokenizer for identifying special tokens (required when freeze_chat_template_tokens=True)
        freeze_chat_template_tokens: Whether to additionally freeze chat template special tokens
        chat_template_freeze_ratio: Ratio of special tokens to freeze (0.0 to 1.0, default: 1.0 = all)
    """
    # Apply the main freezing strategy first
    if strategy == "random_based":
        _freeze_random_parameters(model, freeze_ratio, seed, skip_embeddings_and_head)
    elif strategy == "random_elementwise":
        _freeze_random_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, elementwise=True)
    elif strategy == "random_rowwise":
        _freeze_random_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, elementwise=False, structured_axis='row')
    elif strategy == "hft_based":
        _freeze_hft_parameters(model, freeze_ratio, seed, skip_embeddings_and_head)
    elif strategy == "magnitude_based":
        _freeze_magnitude_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head)
    elif strategy == "magnitude_elementwise":
        _freeze_magnitude_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, elementwise=True)
    elif strategy == "magnitude_rowwise":
        _freeze_magnitude_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, elementwise=False, structured_axis='row')
    elif strategy == "ssu_based":
        _freeze_ssu_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples)
    elif strategy == "ssu_elementwise":
        _freeze_ssu_elementwise_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples)
    elif strategy == "ssu_rowwise":
        _freeze_ssu_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples, axis_preference='row')
    elif strategy == "fisher_based":
        _freeze_fisher_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples, axis_preference='column')
    elif strategy == "fisher_rowwise":
        _freeze_fisher_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples, axis_preference='row')
    elif strategy == "fisher_elementwise":
        _freeze_fisher_elementwise_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples)
    elif strategy == "sgpt_based":
        _freeze_sgpt_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples)
    elif strategy == "sgpt_rowwise":
        _freeze_sgpt_based_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples, axis_preference='row')
    elif strategy == "sgpt_elementwise":
        _freeze_sgpt_elementwise_parameters(model, freeze_ratio, seed, skip_embeddings_and_head, calibration_data, num_calibration_samples)
    else:
        raise ValueError(f"Unknown freezing strategy: {strategy}")
    
    # Apply chat template token freezing as an additional topping if requested
    if freeze_chat_template_tokens:
        if tokenizer is None:
            raise ValueError("Tokenizer is required when freeze_chat_template_tokens=True")
        _freeze_chat_template_tokens_topping(model, tokenizer, chat_template_freeze_ratio, seed)


########## Random freezing


def _freeze_random_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, elementwise=False, structured_axis: Optional[str] = None):
    """
    Randomly freeze a certain ratio of parameters within each module in the model.
    For linear layers, this freezes individual neurons/weights rather than entire modules.
    
    Args:
        model: The model to apply random freezing to
        freeze_ratio: Ratio of parameters to freeze (0.0 to 1.0)
        seed: Random seed for reproducibility
        skip_embeddings_and_head: Whether to skip freezing embedding layers and lm_head
        elementwise: If True, freeze individual elements in 2D matrices; if False, freeze entire rows/columns
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    elementwise_msg = " element-wise" if elementwise else " structured"
    axis_msg = f" ({structured_axis}-wise)" if (not elementwise and structured_axis in ['row','column']) else ""
    print(f"Applying{elementwise_msg}{axis_msg} fine-grained random parameter freezing with ratio: {freeze_ratio}{skip_msg}")
    print(f"Strategy: Randomly freeze {freeze_ratio:.1%} of parameters in each module{skip_msg}")
    
    total_params = 0
    frozen_params = 0
    skipped_params = 0
    modules_processed = 0
    skipped_modules = 0
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Iterate through all named modules
    for name, module in actual_model.named_modules():
        # Skip the root module
        if name == "":
            continue
            
        # Get all parameters in this module (non-recursive to avoid double-counting)
        module_params = list(module.parameters(recurse=False))
        if not module_params:
            continue
        
        # Check if this module should be skipped (By default, skip embeddings and lm_head to follow the original HFT paper)
        if _should_skip_module(name, True):
            module_param_count = sum(p.numel() for p in module_params)
            skipped_params += module_param_count
            total_params += module_param_count
            skipped_modules += 1
            print(f"\nSkipping module '{name}' ({type(module).__name__}): {module_param_count:,} parameters")
            continue
            
        modules_processed += 1
        module_total_params = 0
        module_frozen_params = 0
        
        print(f"\nProcessing module '{name}' ({type(module).__name__})")
        
        for param_idx, param in enumerate(module_params):
            param_name = f"param_{param_idx}"
            if hasattr(module, 'weight') and param is module.weight:
                param_name = "weight"
            elif hasattr(module, 'bias') and param is module.bias:
                param_name = "bias"
            
            # Skip frozen parameters
            if param.requires_grad is False:
                print(f"  Skipping frozen parameter '{param_name}' in module '{name}'")
                continue
            
            param_size = param.numel()
            module_total_params += param_size
            total_params += param_size
            
            # Apply fine-grained freezing based on parameter type and shape
            frozen_count = _freeze_parameter_elements(param, freeze_ratio, param_name, name, elementwise, structured_axis)
            module_frozen_params += frozen_count
            frozen_params += frozen_count
            
            if frozen_count > 0:
                print(f"  {param_name} ({list(param.shape)}): {frozen_count}/{param_size} elements frozen "
                      f"({frozen_count/param_size:.2%})")
        
        if module_total_params > 0:
            print(f"  Module total: {module_frozen_params}/{module_total_params} parameters frozen "
                  f"({module_frozen_params/module_total_params:.2%})")
    
    # Account for lm_head as it is not under model.model
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        lm_head_params = list(model.lm_head.parameters())
        if lm_head_params:
            lm_head_param_count = sum(p.numel() for p in lm_head_params)
            skipped_params += lm_head_param_count
            total_params += lm_head_param_count
            skipped_modules += 1
            print(f"Skipping lm_head: {lm_head_param_count:,} parameters")

    # Store the frozen parameter count in the model for accurate counting
    actual_model._hft_frozen_params = frozen_params
    actual_model._hft_total_params = total_params
    actual_model._hft_freeze_strategy = "random"
    
    print(f"\nRandom freezing completed:")
    print(f"  - Processed {modules_processed} modules")
    if skipped_modules > 0:
        print(f"  - Skipped {skipped_modules} modules (embeddings/lm_head): {skipped_params:,} parameters")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")


def _freeze_parameter_elements(param, freeze_ratio, param_name, module_name, elementwise=False, structured_axis: Optional[str] = None):
    """
    Freeze individual elements within a parameter tensor.
    
    Args:
        param: The parameter tensor to partially freeze
        freeze_ratio: Ratio of elements to freeze
        param_name: Name of the parameter (for logging)
        module_name: Name of the module (for logging)
        elementwise: If True, freeze individual elements; if False, use structured freezing where applicable
    
    Returns:
        int: Number of elements frozen
    """
    if param.numel() == 0:
        return 0
    
    # Create a mask for which elements to freeze
    original_shape = param.shape
    flattened_param = param.view(-1)
    num_elements = flattened_param.numel()
    
    # Determine number of elements to freeze
    num_to_freeze = int(num_elements * freeze_ratio)
    if num_to_freeze == 0:
        return 0
    
    # For structured parameters (like linear layer weights), apply structured or element-wise freezing
    if len(original_shape) == 2 and "weight" in param_name:
        # For 2D weight matrices (like linear layers), freeze entire neurons/features or individual elements
        return _freeze_structured_2d(param, freeze_ratio, original_shape, elementwise, structured_axis)
    elif len(original_shape) == 1:
        # For 1D parameters (like biases), freeze individual elements
        return _freeze_unstructured_1d(param, freeze_ratio)
    else:
        raise NotImplementedError("Freezing not implemented for this parameter shape.")

def _freeze_structured_2d(param, freeze_ratio, shape, elementwise=False, structured_axis: Optional[str] = None):
    """
    Freeze entire rows/columns (structured) or individual elements (element-wise) in a 2D weight matrix.
    
    Args:
        param: The 2D parameter tensor to freeze
        freeze_ratio: Ratio of parameters to freeze (0.0 to 1.0)
        shape: Tuple of (rows, cols) representing the parameter shape
        elementwise: If True, freeze individual elements; if False, freeze entire rows/columns
    
    Returns:
        int: Number of elements frozen
    """
    rows, cols = shape
    
    if elementwise:
        # Element-wise freezing: freeze individual elements randomly
        num_elements = rows * cols
        num_to_freeze = int(num_elements * freeze_ratio)
        if num_to_freeze > 0:
            # Generate random indices for elements to freeze efficiently using torch operations
            # This is much faster than random.sample() for large tensors
            all_indices = torch.randperm(num_elements, device=param.device)[:num_to_freeze]
            frozen_mask = torch.zeros(num_elements, dtype=torch.bool, device=param.device)
            frozen_mask[all_indices] = True
            frozen_mask = frozen_mask.view(shape)
            
            # Store the mask in the parameter for gradient computation
            param.register_hook(_make_mask_hook(frozen_mask))
            
            return num_to_freeze
    else:
        # Structured freezing: freeze entire rows or columns
        axis = structured_axis if structured_axis in ['row', 'column'] else None
        if axis is None:
            # default to columns for transformer MLPs as common practice
            axis = 'column'
        if axis == 'column':
            num_to_freeze = int(cols * freeze_ratio)
            if num_to_freeze > 0:
                cols_to_freeze = random.sample(range(cols), num_to_freeze)
                frozen_mask = torch.zeros_like(param, dtype=torch.bool)
                frozen_mask[:, cols_to_freeze] = True
                param.register_hook(_make_mask_hook(frozen_mask))
                return num_to_freeze * rows
        else:
            num_to_freeze = int(rows * freeze_ratio)
            if num_to_freeze > 0:
                rows_to_freeze = random.sample(range(rows), num_to_freeze)
                frozen_mask = torch.zeros_like(param, dtype=torch.bool)
                frozen_mask[rows_to_freeze, :] = True
                param.register_hook(_make_mask_hook(frozen_mask))
                return num_to_freeze * cols
    
    return 0


def _freeze_unstructured_1d(param, freeze_ratio):
    """
    Freeze random elements in a 1D parameter tensor.
    """
    num_elements = param.numel()
    num_to_freeze = int(num_elements * freeze_ratio)
    
    if num_to_freeze > 0:
        # Use torch operations for efficient random sampling instead of random.sample()
        indices_to_freeze = torch.randperm(num_elements, device=param.device)[:num_to_freeze]
        frozen_mask = torch.zeros_like(param, dtype=torch.bool)
        frozen_mask[indices_to_freeze] = True
        
        # Store the mask in the parameter for gradient computation
        param.register_hook(_make_mask_hook(frozen_mask))
        
        return num_to_freeze
    
    return 0


########## HFT-based freezing


def _freeze_hft_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False):
    """
    Freeze modules using a structured, layer-based strategy. For each transformer layer,
    dynamically choose how many attention (q/k/v/o), FFN (up/gate/down), and norm
    (input/post-attention) modules to freeze so the fraction of frozen modules per layer
    matches the requested `freeze_ratio`.

    Special case: when `freeze_ratio == 0.5`, this reproduces the original config:
      - Freeze 2/4 attention modules
      - Half the layers freeze 2 FFN matrices, the other half freeze 1 (avg 1.5/3)
      - Freeze 1/2 RMSNorm modules

    Args:
        model: The model to apply random freezing to
        freeze_ratio: Target fraction of modules frozen per layer (0.0–1.0)
        seed: Random seed for reproducibility
        skip_embeddings_and_head: Whether to skip freezing embedding layers and lm_head
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    print(f"Applying layer-based module freezing with ratio: {freeze_ratio}{skip_msg}")
    if abs(freeze_ratio - 0.5) < 1e-8:
        print("Strategy: Per layer - 2/4 attention, half layers get 2 feed-forward matrices, half get 1, 1/2 RMSNorm modules (original config)")
    else:
        print("Strategy: Per layer - proportionally freeze attention/FFN/norm modules to match requested ratio")
    
    total_params = 0
    frozen_params = 0
    layers_processed = 0
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Find transformer layers
    layers = _find_transformer_layers(actual_model)
    
    if layers is None:
        print("Warning: Could not find transformer layers. Falling back to basic module freezing.")
        return _freeze_hft_parameters_module_level_basic(model, freeze_ratio, seed, skip_embeddings_and_head)
    
    print(f"Found {len(layers)} transformer layers")
    
    # Prepare FFN assignment sets for the 0.5 special-case
    total_layers = len(layers)
    layers_with_two_matrices = set()
    layers_with_one_matrix = set()
    if abs(freeze_ratio - 0.5) < 1e-8 and total_layers > 0:
        half_layers = total_layers // 2
        layer_indices = list(range(total_layers))
        random.shuffle(layer_indices)
        layers_with_two_matrices = set(layer_indices[:half_layers])
        layers_with_one_matrix = set(layer_indices[half_layers:])
        print(f"Layers freezing 2 feed-forward matrices: {sorted(layers_with_two_matrices)}")
        print(f"Layers freezing 1 feed-forward matrix: {sorted(layers_with_one_matrix)}")
    
    # Process each transformer layer
    for layer_idx, layer in enumerate(layers):
        print(f"\nProcessing Layer {layer_idx}:")
        layer_frozen = 0
        layer_total = 0
        
        # Find attention and MLP modules within this layer
        attention_modules = _find_attention_modules(layer, layer_idx)
        mlp_modules = _find_mlp_modules(layer, layer_idx)
        norm_modules = _find_norm_modules(layer, layer_idx)
        
        # Count layer parameters
        for param in layer.parameters():
            layer_total += param.numel()
        
        # Compute per-category counts to freeze
        n_att = len(attention_modules)
        n_ffn = len(mlp_modules)
        n_norm = len(norm_modules)
        n_total = n_att + n_ffn + n_norm

        att_freeze = 0
        ffn_freeze = 0
        norm_freeze = 0

        if n_total > 0:
            if abs(freeze_ratio - 0.5) < 1e-8:
                # Original behavior at 0.5
                att_freeze = min(2, n_att) if n_att > 0 else 0
                if n_ffn > 0:
                    ffn_freeze = min(2, n_ffn) if layer_idx in layers_with_two_matrices else min(1, n_ffn)
                norm_freeze = 1 if n_norm > 0 else 0
            else:
                # Parameter-count-aware, count-guided selection:
                # 1) Initialize per-category desired counts similar to 50% setting (scaled by ratio)
                # 2) Choose those modules (smallest-first if r<=0.5 else largest-first)
                # 3) Adjust by adding/removing modules across categories to hit param target closely

                # Build candidates with parameter counts per module
                att_cands = []  # (name, module, pc)
                ffn_cands = []
                norm_cands = []
                for name, module in attention_modules:
                    pc = sum(p.numel() for p in module.parameters(recurse=False))
                    if pc > 0:
                        att_cands.append((name, module, pc))
                for name, module in mlp_modules:
                    pc = sum(p.numel() for p in module.parameters(recurse=False))
                    if pc > 0:
                        ffn_cands.append((name, module, pc))
                for name, module in norm_modules:
                    pc = sum(p.numel() for p in module.parameters(recurse=False))
                    if pc > 0:
                        norm_cands.append((name, module, pc))

                layer_total_params = sum(p.numel() for p in layer.parameters())
                target = int(round(freeze_ratio * layer_total_params))
                ascending = freeze_ratio <= 0.5

                # Desired counts per category (rounded ratio of availability)
                att_target = min(len(att_cands), int(round(freeze_ratio * len(att_cands))))
                ffn_target = min(len(ffn_cands), int(round(freeze_ratio * len(ffn_cands))))
                norm_target = min(len(norm_cands), int(round(freeze_ratio * len(norm_cands))))

                # Pick initial selections per category
                keyfn = (lambda x: x[2])
                att_sorted = sorted(att_cands, key=keyfn, reverse=not ascending)
                ffn_sorted = sorted(ffn_cands, key=keyfn, reverse=not ascending)
                norm_sorted = sorted(norm_cands, key=keyfn, reverse=not ascending)

                att_sel = att_sorted[:att_target]
                ffn_sel = ffn_sorted[:ffn_target]
                norm_sel = norm_sorted[:norm_target]

                selected = [('att', *t) for t in att_sel] + [('ffn', *t) for t in ffn_sel] + [('norm', *t) for t in norm_sel]
                cum = sum(t[3] for t in selected)

                # Pools for remaining candidates
                att_rem = [c for c in att_sorted if c not in att_sel]
                ffn_rem = [c for c in ffn_sorted if c not in ffn_sel]
                norm_rem = [c for c in norm_sorted if c not in norm_sel]
                rem_all = [('att', *t) for t in att_rem] + [('ffn', *t) for t in ffn_rem] + [('norm', *t) for t in norm_rem]
                rem_all = sorted(rem_all, key=lambda x: x[3], reverse=not ascending)

                # Adjust up by adding candidates to approach the target
                i = 0
                while cum < target and i < len(rem_all):
                    grp, nm, mod, pc = rem_all[i]
                    # Add if it improves closeness
                    if abs(target - (cum + pc)) <= abs(target - cum):
                        selected.append((grp, nm, mod, pc))
                        cum += pc
                    i += 1

                # If overshot, try removing one selected module that best improves closeness
                if cum > target and selected:
                    best_idx = None
                    best_delta = abs(cum - target)
                    for idx, s in enumerate(selected):
                        delta = abs((cum - s[3]) - target)
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = idx
                    if best_idx is not None:
                        grp, nm, mod, pc = selected.pop(best_idx)
                        cum -= pc

                # Map selections back to per-category counts
                att_freeze = sum(1 for s in selected if s[0] == 'att')
                ffn_freeze = sum(1 for s in selected if s[0] == 'ffn')
                norm_freeze = sum(1 for s in selected if s[0] == 'norm')

        # Freeze attention modules
        if n_att > 0 and att_freeze > 0:
            modules_to_freeze = random.sample(attention_modules, att_freeze)
            for name, module in modules_to_freeze:
                frozen_count = _freeze_module_completely(module)
                layer_frozen += frozen_count
                print(f"  Froze attention module '{name}': {frozen_count:,} parameters")

        # Freeze FFN modules
        if n_ffn > 0 and ffn_freeze > 0:
            modules_to_freeze = random.sample(mlp_modules, ffn_freeze)
            print(f"  Layer {layer_idx}: freezing {ffn_freeze} feed-forward matrix/matrices")
            for name, module in modules_to_freeze:
                frozen_count = _freeze_module_completely(module)
                layer_frozen += frozen_count
                print(f"  Froze MLP module '{name}': {frozen_count:,} parameters")

        # Freeze norm modules
        if n_norm > 0 and norm_freeze > 0:
            modules_to_freeze = random.sample(norm_modules, norm_freeze)
            for name, module in modules_to_freeze:
                frozen_count = _freeze_module_completely(module)
                layer_frozen += frozen_count
                print(f"  Froze norm module '{name}': {frozen_count:,} parameters")
        
        frozen_params += layer_frozen
        total_params += layer_total
        layers_processed += 1
        
        if layer_total > 0:
            print(f"  Layer {layer_idx} total: {layer_frozen}/{layer_total} parameters frozen ({layer_frozen/layer_total:.2%})")
    
    print(f"\nHFT freezing completed:")
    print(f"  - Processed {layers_processed} transformer layers")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")


def _find_transformer_layers(model):
    """Find transformer layers in the model."""
    # Try different layer access patterns for different architectures
    layer_paths = [
        'layers',           # Most common (LLaMA, Qwen, etc.)
        'transformer.h',    # GPT-2, GPT-Neo
        'transformer.layers', # Some transformer variants
        'h',               # Some GPT variants
        'decoder.layers',   # Some decoder-only models
        'encoder.layers',   # Some encoder models
        'blocks',          # Some transformer variants
        'model.layers',    # Nested model structure
    ]
    
    for path in layer_paths:
        try:
            obj = model
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if hasattr(obj, '__len__') and hasattr(obj, '__getitem__'):
                print(f"Found transformer layers at: {path}")
                return obj
        except AttributeError:
            continue
    
    return None


def _find_attention_modules(layer, layer_idx):
    """Find attention-related modules in a transformer layer."""
    attention_modules = []
    
    # Common attention module names
    attention_patterns = [
        ('q_proj', ['q_proj', 'query', 'to_q']),
        ('k_proj', ['k_proj', 'key', 'to_k']), 
        ('v_proj', ['v_proj', 'value', 'to_v']),
        ('o_proj', ['o_proj', 'out_proj', 'to_out', 'dense'])
    ]
    
    for module_type, patterns in attention_patterns:
        for name, module in layer.named_modules():
            if name and any(pattern in name.lower() for pattern in patterns):
                # Make sure this is a leaf module with parameters
                if list(module.parameters(recurse=False)):
                    attention_modules.append((f"layer_{layer_idx}.{name}", module))
                    break
    
    return attention_modules


def _find_mlp_modules(layer, layer_idx):
    """Find MLP/FFN-related modules in a transformer layer."""
    mlp_modules = []
    
    # Common MLP module names
    mlp_patterns = [
        ('up_proj', ['up_proj', 'fc1', 'w1', 'wi_0']),
        ('down_proj', ['down_proj', 'fc2', 'w2', 'wo']),
        ('gate_proj', ['gate_proj', 'w3', 'wi_1'])
    ]
    
    for module_type, patterns in mlp_patterns:
        for name, module in layer.named_modules():
            if name and any(pattern in name.lower() for pattern in patterns):
                # Make sure this is a leaf module with parameters
                if list(module.parameters(recurse=False)):
                    mlp_modules.append((f"layer_{layer_idx}.{name}", module))
                    break
    
    return mlp_modules


def _find_norm_modules(layer, layer_idx):
    """Find normalization modules in a transformer layer."""
    norm_modules = []
    
    # Common norm module names
    norm_patterns = [
        ('input_layernorm', ['input_layernorm', 'ln_1', 'norm1', 'attention_norm']),
        ('post_attention_layernorm', ['post_attention_layernorm', 'ln_2', 'norm2', 'ffn_norm'])
    ]
    
    for module_type, patterns in norm_patterns:
        for name, module in layer.named_modules():
            if name and any(pattern in name.lower() for pattern in patterns):
                # Make sure this is a leaf module with parameters
                if list(module.parameters(recurse=False)):
                    norm_modules.append((f"layer_{layer_idx}.{name}", module))
                    break
    
    return norm_modules


def _freeze_module_completely(module):
    """Freeze all parameters in a module and return count of frozen parameters."""
    frozen_count = 0
    for param in module.parameters(recurse=False):
        param.requires_grad = False
        frozen_count += param.numel()
    return frozen_count


def _freeze_hft_parameters_module_level_basic(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False):
    """
    Fallback basic module-level freezing when layer structure cannot be determined.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    print(f"Applying basic module-level random parameter freezing with ratio: {freeze_ratio}{skip_msg}")
    
    total_params = 0
    frozen_params = 0
    skipped_params = 0
    modules_processed = 0
    skipped_modules = 0
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Define module types that should be treated as units
    layer_keywords = ['layer', 'block', 'transformer_block']
    attention_keywords = ['attention', 'attn', 'self_attn', 'cross_attn']
    ffn_keywords = ['mlp', 'ffn', 'feed_forward', 'fc']
    embedding_keywords = ['embed', 'embedding', 'wte', 'wpe']
    
    # Collect modules by type for hierarchical freezing
    layer_modules = []
    attention_modules = []
    ffn_modules = []
    embedding_modules = []
    other_modules = []
    
    for name, module in actual_model.named_modules():
        # Skip the root module and modules without parameters
        if name == "" or not list(module.parameters(recurse=False)):
            continue
        
        # Check if this module should be skipped
        if _should_skip_module(name, skip_embeddings_and_head):
            module_param_count = sum(p.numel() for p in module.parameters(recurse=False))
            skipped_params += module_param_count
            skipped_modules += 1
            print(f"Skipping module '{name}': {module_param_count:,} parameters")
            continue
            
        name_lower = name.lower()
        
        # Categorize modules
        if any(keyword in name_lower for keyword in layer_keywords):
            layer_modules.append((name, module))
        elif any(keyword in name_lower for keyword in attention_keywords):
            attention_modules.append((name, module))
        elif any(keyword in name_lower for keyword in ffn_keywords):
            ffn_modules.append((name, module))
        elif any(keyword in name_lower for keyword in embedding_keywords):
            embedding_modules.append((name, module))
        else:
            other_modules.append((name, module))
    
    # Apply freezing to each category
    all_module_groups = [
        ("Layer modules", layer_modules),
        ("Attention modules", attention_modules),
        ("FFN modules", ffn_modules),
        ("Embedding modules", embedding_modules),
        ("Other modules", other_modules)
    ]
    
    for group_name, module_list in all_module_groups:
        if not module_list:
            continue
            
        print(f"\nProcessing {group_name}: {len(module_list)} modules")
        
        # Determine how many modules to freeze in this group
        num_modules_to_freeze = max(1, int(len(module_list) * freeze_ratio))
        
        # Randomly select modules to freeze
        modules_to_freeze = random.sample(module_list, num_modules_to_freeze)
        
        for name, module in modules_to_freeze:
            module_params = list(module.parameters(recurse=False))
            module_param_count = sum(p.numel() for p in module_params)
            
            # Freeze all parameters in this module
            for param in module_params:
                param.requires_grad = False
            
            frozen_params += module_param_count
            modules_processed += 1
            
            print(f"  Froze module '{name}': {module_param_count:,} parameters")
    
    # Count total parameters
    for param in actual_model.parameters():
        total_params += param.numel()
    
    print(f"\nBasic module-level random freezing completed:")
    print(f"  - Processed {modules_processed} modules")
    if skipped_modules > 0:
        print(f"  - Skipped {skipped_modules} modules (embeddings/lm_head): {skipped_params:,} parameters")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")

def _freeze_hft_parameters_module_level_basic(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False):
    """
    Fallback basic module-level freezing when layer structure cannot be determined.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    print(f"Applying basic module-level random parameter freezing with ratio: {freeze_ratio}{skip_msg}")
    
    total_params = 0
    frozen_params = 0
    skipped_params = 0
    modules_processed = 0
    skipped_modules = 0
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Define module types that should be treated as units
    layer_keywords = ['layer', 'block', 'transformer_block']
    attention_keywords = ['attention', 'attn', 'self_attn', 'cross_attn']
    ffn_keywords = ['mlp', 'ffn', 'feed_forward', 'fc']
    embedding_keywords = ['embed', 'embedding', 'wte', 'wpe']
    
    # Collect modules by type for hierarchical freezing
    layer_modules = []
    attention_modules = []
    ffn_modules = []
    embedding_modules = []
    other_modules = []
    
    for name, module in actual_model.named_modules():
        # Skip the root module and modules without parameters
        if name == "" or not list(module.parameters(recurse=False)):
            continue
        
        # Check if this module should be skipped
        if _should_skip_module(name, skip_embeddings_and_head):
            module_param_count = sum(p.numel() for p in module.parameters(recurse=False))
            skipped_params += module_param_count
            skipped_modules += 1
            print(f"Skipping module '{name}': {module_param_count:,} parameters")
            continue
            
        name_lower = name.lower()
        
        # Categorize modules
        if any(keyword in name_lower for keyword in layer_keywords):
            layer_modules.append((name, module))
        elif any(keyword in name_lower for keyword in attention_keywords):
            attention_modules.append((name, module))
        elif any(keyword in name_lower for keyword in ffn_keywords):
            ffn_modules.append((name, module))
        elif any(keyword in name_lower for keyword in embedding_keywords):
            embedding_modules.append((name, module))
        else:
            other_modules.append((name, module))
    
    # Apply freezing to each category
    all_module_groups = [
        ("Layer modules", layer_modules),
        ("Attention modules", attention_modules),
        ("FFN modules", ffn_modules),
        ("Embedding modules", embedding_modules),
        ("Other modules", other_modules)
    ]
    
    for group_name, module_list in all_module_groups:
        if not module_list:
            continue
            
        print(f"\nProcessing {group_name}: {len(module_list)} modules")
        
        # Determine how many modules to freeze in this group
        num_modules_to_freeze = max(1, int(len(module_list) * freeze_ratio))
        
        # Randomly select modules to freeze
        modules_to_freeze = random.sample(module_list, num_modules_to_freeze)
        
        for name, module in modules_to_freeze:
            module_params = list(module.parameters(recurse=False))
            module_param_count = sum(p.numel() for p in module_params)
            
            # Freeze all parameters in this module
            for param in module_params:
                param.requires_grad = False
            
            frozen_params += module_param_count
            modules_processed += 1
            
            print(f"  Froze module '{name}': {module_param_count:,} parameters")
    
    # Count total parameters
    for param in actual_model.parameters():
        total_params += param.numel()
    
    print(f"\nBasic HFT random freezing completed:")
    print(f"  - Processed {modules_processed} modules")
    if skipped_modules > 0:
        print(f"  - Skipped {skipped_modules} modules (embeddings/lm_head): {skipped_params:,} parameters")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")


########## Magnitude-based freezing


def _freeze_magnitude_based_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, elementwise=False, structured_axis: Optional[str] = None):
    """
    Freeze parameters based on their magnitude. Large magnitude parameters (important for 
    current knowledge) are frozen to prevent catastrophic forgetting, while small magnitude 
    parameters remain trainable to allow adaptation.
    
    Args:
        model: The model to apply magnitude-based freezing to
        freeze_ratio: Ratio of parameters to freeze (0.0 to 1.0) - freezes largest magnitude params
        seed: Random seed for reproducibility (used for tie-breaking)
        skip_embeddings_and_head: Whether to skip freezing embedding layers and lm_head
        elementwise: If True, freeze individual elements in 2D matrices; if False, freeze entire rows/columns
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    elementwise_msg = " element-wise" if elementwise else " structured"
    axis_msg = f" ({structured_axis}-wise)" if (not elementwise and structured_axis in ['row','column']) else ""
    print(f"Applying{elementwise_msg}{axis_msg} magnitude-based parameter freezing with ratio: {freeze_ratio}{skip_msg}")
    print("Strategy: Freeze large magnitude weights (preserve knowledge), train small magnitude weights (allow adaptation)")
    
    total_params = 0
    frozen_params = 0
    skipped_params = 0
    modules_processed = 0
    skipped_modules = 0
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Iterate through all named modules
    for name, module in actual_model.named_modules():
        # Skip the root module
        if name == "":
            continue
            
        # Get all parameters in this module (non-recursive to avoid double-counting)
        module_params = list(module.parameters(recurse=False))
        if not module_params:
            continue
        
        # Check if this module should be skipped
        if _should_skip_module(name, True): # By default, we should skip embeddings and lm_head to follow the original HFT paper
            module_param_count = sum(p.numel() for p in module_params)
            skipped_params += module_param_count
            total_params += module_param_count
            skipped_modules += 1
            print(f"\nSkipping module '{name}' ({type(module).__name__}): {module_param_count:,} parameters")
            continue
            
        modules_processed += 1
        module_total_params = 0
        module_frozen_params = 0
        
        print(f"\nProcessing module '{name}' ({type(module).__name__})")
        
        for param_idx, param in enumerate(module_params):
            param_name = f"param_{param_idx}"
            if hasattr(module, 'weight') and param is module.weight:
                param_name = "weight"
            elif hasattr(module, 'bias') and param is module.bias:
                param_name = "bias"
            
            # Skip frozen parameters
            if param.requires_grad is False:
                print(f"  Skipping frozen parameter '{param_name}' in module '{name}'")
                continue
            
            param_size = param.numel()
            module_total_params += param_size
            total_params += param_size
            
            # Apply magnitude-based freezing
            frozen_count = _freeze_by_magnitude(param, freeze_ratio, param_name, name, elementwise, structured_axis)
            module_frozen_params += frozen_count
            frozen_params += frozen_count
            
            if frozen_count > 0:
                frozen_percentage = frozen_count / param_size
                print(f"  {param_name} ({list(param.shape)}): {frozen_count}/{param_size} elements frozen "
                      f"({frozen_percentage:.2%}) - large magnitude weights")
        
        if module_total_params > 0:
            print(f"  Module total: {module_frozen_params}/{module_total_params} parameters frozen "
                  f"({module_frozen_params/module_total_params:.2%})")
    
    # Account for lm_head as it is not under model.model
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        lm_head_params = list(model.lm_head.parameters())
        if lm_head_params:
            lm_head_param_count = sum(p.numel() for p in lm_head_params)
            skipped_params += lm_head_param_count
            total_params += lm_head_param_count
            skipped_modules += 1
            print(f"Skipping lm_head: {lm_head_param_count:,} parameters")
    
    # Store the frozen parameter count in the model for accurate counting
    actual_model._hft_frozen_params = frozen_params
    actual_model._hft_total_params = total_params
    actual_model._hft_freeze_strategy = "magnitude"
    
    print(f"\nMagnitude-based freezing completed:")
    print(f"  - Processed {modules_processed} modules")
    if skipped_modules > 0:
        print(f"  - Skipped {skipped_modules} modules (embeddings/lm_head): {skipped_params:,} parameters")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")
    print(f"  - Strategy: Large magnitude weights frozen (preserve knowledge)")
    print(f"  - Strategy: Small magnitude weights trainable (allow adaptation)")


def _freeze_by_magnitude(param, freeze_ratio, param_name, module_name, elementwise=False, structured_axis: Optional[str] = None):
    """
    Freeze parameters based on their absolute magnitude values.
    Freezes the largest magnitude parameters to preserve important knowledge.
    
    Args:
        param: The parameter tensor to apply magnitude-based freezing to
        freeze_ratio: Ratio of parameters to freeze (largest magnitude ones)
        param_name: Name of the parameter (for logging)
        module_name: Name of the module (for logging)
        elementwise: If True, freeze individual elements; if False, use structured freezing where applicable
    
    Returns:
        int: Number of elements frozen
    """
    if param.numel() == 0:
        return 0
    
    # Work with absolute magnitudes to identify important parameters
    abs_param = torch.abs(param)
    original_shape = param.shape
    
    # Flatten for easier processing
    flat_param = abs_param.view(-1)
    num_elements = flat_param.numel()
    
    # Determine number of elements to freeze (largest magnitude ones)
    num_to_freeze = int(num_elements * freeze_ratio)
    if num_to_freeze == 0:
        return 0
    
    # For structured parameters (like linear layer weights), apply structured or element-wise magnitude-based freezing
    if len(original_shape) == 2 and "weight" in param_name:
        return _freeze_structured_magnitude_2d(param, freeze_ratio, original_shape, elementwise, structured_axis)
    else:
        # For other parameters, use element-wise magnitude-based freezing
        return _freeze_unstructured_magnitude(param, freeze_ratio, original_shape)


def _freeze_structured_magnitude_2d(param, freeze_ratio, shape, elementwise=False, structured_axis: Optional[str] = None):
    """
    Freeze entire rows/columns (structured) or individual elements (element-wise) in a 2D weight matrix based on their magnitude.
    This preserves the structured nature of linear transformations when elementwise=False.
    
    Args:
        param: The 2D parameter tensor to freeze
        freeze_ratio: Ratio of parameters to freeze (largest magnitude ones)
        shape: Tuple of (rows, cols) representing the parameter shape
        elementwise: If True, freeze individual elements; if False, freeze entire rows/columns
    
    Returns:
        int: Number of elements frozen
    """
    rows, cols = shape
    abs_param = torch.abs(param)
    
    if elementwise:
        # Element-wise magnitude-based freezing: freeze individual elements with largest magnitudes
        flat_param = abs_param.view(-1)
        num_elements = flat_param.numel()
        num_to_freeze = int(num_elements * freeze_ratio)
        
        if num_to_freeze > 0:
            # Get indices of largest magnitude elements
            _, largest_indices = torch.topk(flat_param, num_to_freeze, largest=True)
            
            # Create mask in original shape
            frozen_mask = torch.zeros(num_elements, dtype=torch.bool, device=param.device)
            frozen_mask[largest_indices] = True
            frozen_mask = frozen_mask.view(shape)
            
            # Store the mask in the parameter for gradient computation
            param.register_hook(_make_mask_hook(frozen_mask))
            
            # Log statistics
            frozen_magnitudes = flat_param[largest_indices]
            avg_frozen_magnitude = frozen_magnitudes.mean().item()
            min_frozen_magnitude = frozen_magnitudes.min().item()
            max_frozen_magnitude = frozen_magnitudes.max().item()
            
            print(f"    Froze {num_to_freeze} elements with magnitude range: [{min_frozen_magnitude:.6f}, {max_frozen_magnitude:.6f}], avg: {avg_frozen_magnitude:.6f}")
            
            return num_to_freeze
    else:
        # Structured magnitude-based freezing: freeze entire rows or columns based on their magnitude
        # Calculate row/column magnitudes (L2 norm)
        row_magnitudes = torch.norm(abs_param, dim=1)  # [rows]
        col_magnitudes = torch.norm(abs_param, dim=0)  # [cols]

        # Decide axis
        axis = structured_axis if structured_axis in ['row', 'column'] else "column" # Default to column-wise
        
        if axis == 'column':
            num_to_freeze = int(cols * freeze_ratio)
            if num_to_freeze > 0:
                _, largest_indices = torch.topk(col_magnitudes, num_to_freeze, largest=True)
                frozen_mask = torch.zeros_like(param, dtype=torch.bool)
                frozen_mask[:, largest_indices] = True
                param.register_hook(_make_mask_hook(frozen_mask))
                avg_frozen_magnitude = col_magnitudes[largest_indices].mean().item()
                print(f"    Froze {num_to_freeze} columns (input features) with avg magnitude: {avg_frozen_magnitude:.6f}")
                return num_to_freeze * rows
        else:
            num_to_freeze = int(rows * freeze_ratio)
            if num_to_freeze > 0:
                _, largest_indices = torch.topk(row_magnitudes, num_to_freeze, largest=True)
                frozen_mask = torch.zeros_like(param, dtype=torch.bool)
                frozen_mask[largest_indices, :] = True
                param.register_hook(_make_mask_hook(frozen_mask))
                avg_frozen_magnitude = row_magnitudes[largest_indices].mean().item()
                print(f"    Froze {num_to_freeze} rows (output neurons) with avg magnitude: {avg_frozen_magnitude:.6f}")
                return num_to_freeze * cols
    
    return 0


def _freeze_unstructured_magnitude(param, freeze_ratio, original_shape):
    """
    Freeze individual elements based on their magnitude values.
    """
    abs_param = torch.abs(param)
    flat_param = abs_param.view(-1)
    num_elements = flat_param.numel()
    num_to_freeze = int(num_elements * freeze_ratio)
    
    if num_to_freeze > 0:
        # Get indices of largest magnitude elements
        _, largest_indices = torch.topk(flat_param, num_to_freeze, largest=True)
        
        # Create mask in original shape
        frozen_mask = torch.zeros(num_elements, dtype=torch.bool, device=param.device)
        frozen_mask[largest_indices] = True
        frozen_mask = frozen_mask.view(original_shape)
        
        # Store the mask in the parameter for gradient computation
        param.register_hook(_make_mask_hook(frozen_mask))
        
        # Log statistics
        frozen_magnitudes = flat_param[largest_indices]
        avg_frozen_magnitude = frozen_magnitudes.mean().item()
        min_frozen_magnitude = frozen_magnitudes.min().item()
        max_frozen_magnitude = frozen_magnitudes.max().item()
        
        print(f"    Froze elements with magnitude range: [{min_frozen_magnitude:.6f}, {max_frozen_magnitude:.6f}], avg: {avg_frozen_magnitude:.6f}")
        
        return num_to_freeze
    
    return 0


########## SSU-based freezing


def _collect_activation_statistics(model, calibration_data, num_samples=128):
    """
    Collect activation statistics from calibration data to compute SSU scores.
    
    Args:
        model: The model to collect activations from
        calibration_data: DataLoader or iterable with calibration samples
        num_samples: Number of samples to use for calibration
    
    Returns:
        dict: Dictionary mapping parameter names to activation importance statistics
    """
    model.eval()
    activation_stats = {}
    hooks = []
    
    # Hook function to collect input activations (Wanda only uses input activations)
    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {
                    'input_activations': []
                }

            # Only use floating-point inputs; skip if inputs are integer (e.g., token ids)
            if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor) and input[0].is_floating_point():
                act = input[0].detach()
                # Compute activation importance as squared mean over all dims except last (feature dim)
                if act.dim() >= 2:
                    reduce_dims = list(range(act.dim() - 1))
                    input_importance = (act ** 2).mean(dim=reduce_dims)
                else:
                    # 1D vector: use squared values directly as importance
                    input_importance = (act ** 2)
                activation_stats[name]['input_activations'].append(input_importance.cpu())

        return hook_fn
    
    # Register hooks on all modules with parameters
    actual_model = model.model if hasattr(model, 'model') else model
    for name, module in actual_model.named_modules():
        if not list(module.parameters(recurse=False)):
            continue
        lower_name = name.lower()
        # Skip embeddings and lm_head modules for activation collection
        try:
            is_embedding = isinstance(module, torch.nn.Embedding) or 'embedding' in lower_name or 'embed' in lower_name
        except Exception:
            is_embedding = 'embedding' in lower_name or 'embed' in lower_name
        is_lm_head = 'lm_head' in lower_name or lower_name.endswith('.lm_head') or name == 'lm_head'
        if is_embedding or is_lm_head:
            continue
        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)
    
    # Collect activation statistics
    sample_count = 0
    try:
        with torch.no_grad():
            for batch in calibration_data:
                if sample_count >= num_samples:
                    break

                # Normalize batch into model input(s)
                from collections.abc import Mapping
                device = next(actual_model.parameters()).device

                model_inputs = None
                batch_size = 1

                if isinstance(batch, Mapping):
                    # Typical HF BatchEncoding/dict: pick tensor-like entries and move to device
                    model_inputs = {
                        k: (v.to(device) if hasattr(v, 'to') else v)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    # Determine batch size from input_ids if present, else from first tensor value
                    bs_source = model_inputs.get('input_ids', None)
                    if bs_source is None and len(model_inputs) > 0:
                        bs_source = next(iter(model_inputs.values()))
                    if isinstance(bs_source, torch.Tensor) and bs_source.dim() > 0:
                        batch_size = bs_source.shape[0]
                elif isinstance(batch, (list, tuple)):
                    # Assume first element is input_ids or a tensor dict
                    first = batch[0]
                    if isinstance(first, Mapping):
                        model_inputs = {
                            k: (v.to(device) if hasattr(v, 'to') else v)
                            for k, v in first.items()
                            if isinstance(v, torch.Tensor)
                        }
                        bs_source = model_inputs.get('input_ids', None)
                        if bs_source is None and len(model_inputs) > 0:
                            bs_source = next(iter(model_inputs.values()))
                        if isinstance(bs_source, torch.Tensor) and bs_source.dim() > 0:
                            batch_size = bs_source.shape[0]
                    else:
                        # Treat as tensor inputs
                        inputs = first
                        if hasattr(inputs, 'to'):
                            inputs = inputs.to(device)
                        if hasattr(inputs, 'shape') and len(inputs.shape) > 0:
                            batch_size = inputs.shape[0]
                        # Call model with positional tensor as input_ids
                        _ = model(input_ids=inputs)
                        sample_count += batch_size
                        if sample_count >= num_samples:
                            break
                        continue
                else:
                    # Single tensor
                    inputs = batch
                    if hasattr(inputs, 'to'):
                        inputs = inputs.to(device)
                    if hasattr(inputs, 'shape') and len(inputs.shape) > 0:
                        batch_size = inputs.shape[0]
                    _ = model(input_ids=inputs)
                    sample_count += batch_size
                    if sample_count >= num_samples:
                        break
                    continue

                # Forward pass to collect activations
                if model_inputs is not None and len(model_inputs) > 0:
                    _ = model(**model_inputs)
                else:
                    # As a fallback, do nothing this iteration
                    pass

                sample_count += batch_size
                if sample_count >= num_samples:
                    break
    
    except Exception as e:
        print(f"Warning: Error during activation collection: {e}")
        print("Falling back to weight-based importance scoring")
        return None
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Process collected statistics
    processed_stats = {}
    for module_name, stats in activation_stats.items():
        if stats['input_activations']:
            # Average input activation importance across all calibration samples
            avg_input_importance = torch.stack(stats['input_activations']).mean(dim=0)
            
            # Create activation importance object for Wanda scoring
            class ActivationImportance:
                def __init__(self, input_act):
                    self.input_activations = input_act
                    # For Wanda, we primarily use input activations
                    # Create element-wise importance for unstructured parameters
                    self.element_importance = input_act.view(-1)
            
            # Store stats for both weight and bias parameters
            processed_stats[f"{module_name}.weight"] = ActivationImportance(avg_input_importance)
            processed_stats[f"{module_name}.bias"] = ActivationImportance(avg_input_importance)
    
    print(f"Collected activation statistics for {len(processed_stats)} parameters from {sample_count} samples")
    return processed_stats


# ============================================================
# LA-SSU: Language-aware Soft Source-Shielded Updates
# ============================================================
#
# 主方法设计：
#   复用官方 SSU / Wanda 的 column-wise 重要性分数。
#   不改变“怎么打分”，只改变“打分之后如何更新”。
#
# 原始 SSU:
#   source calibration data
#   -> Wanda score
#   -> 每个矩阵选 top-k 重要 columns
#   -> 这些 columns 的梯度置 0，也就是 hard freeze
#
# LA-SSU:
#   每种语言分别用同一套 SSU/Wanda 方法计算重要 columns
#   -> 保存每种语言的 top columns
#   -> 根据历史语言和当前语言的 overlap 分类
#   -> 不同类别使用不同 update scale
#
# 例如：
#   Old-Specific columns            scale = 0.0
#   Old-Shared columns              scale = 0.1
#   Current-History-Shared columns  scale = 0.5
#   Current-Specific columns        scale = 1.0
#   Others                          scale = 1.0
#
# 注意：
#   这是 column-wise soft SSU，不是 FFN-neuron 版本。
#   它最接近官方 SSU，只是把 binary mask 扩展成 soft scale mask。


def _lassu_get_actual_model(model):
    """
    兼容 HuggingFace CausalLM 包装结构。

    有些模型是：
        model.layers
    有些模型是：
        model.model.layers

    所以这里统一取内部 actual_model。
    """
    return model.model if hasattr(model, "model") else model


def _lassu_score_file_load(path):
    """
    读取 LA-SSU 保存的 score 文件。

    文件格式：
        {
            "__meta__": {...},
            "layers.0.mlp.down_proj.weight": {
                "scores": Tensor[num_units],
                "top_idx": Tensor[top_k],
                "num_units": int,
                "unit_axis": "column" or "element",
                "param_shape": tuple,
                ...
            },
            ...
        }
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"[LA-SSU] score file should be a dict, got {type(obj)} from {path}")
    return obj


def _lassu_lookup_param(param_map, full_name):
    """
    根据参数名查找参数。

    为什么需要这个函数？
        score 文件里保存的名字可能是：
            layers.0.mlp.down_proj.weight
        但模型 named_parameters() 里可能是：
            model.layers.0.mlp.down_proj.weight

    所以这里尝试几种常见前缀。
    """
    candidates = [full_name]

    for prefix in ["model.", "module."]:
        if full_name.startswith(prefix):
            candidates.append(full_name[len(prefix):])
        else:
            candidates.append(prefix + full_name)

    for cand in candidates:
        if cand in param_map:
            return cand, param_map[cand]

    return None, None


def _lassu_compute_unit_scores(param, activation_importance=None):
    """
    计算一个参数的 LA-SSU unit score。

    对 2D weight:
        param shape = [out_features, in_features]
        unit = column
        第 j 个 column 表示第 j 个 input feature pathway。
        这和官方 SSU 的 column-wise 逻辑一致。

        score_j = || W[:, j] ||_2 * input_activation_importance[j]

    对 1D 参数:
        论文里说 1D 参数每个 element 可以看成自己的 column。
        所以 unit = element。

    返回：
        scores: Tensor[num_units]
        unit_axis: "column" or "element"
    """
    with torch.no_grad():
        abs_param = param.detach().abs()

        # 2D Linear weight: column-wise score
        if param.dim() == 2:
            rows, cols = param.shape

            # 默认先用权重范数作为 fallback
            col_magnitudes = torch.norm(abs_param, dim=0)  # [cols]
            scores = col_magnitudes

            if activation_importance is not None and hasattr(activation_importance, "input_activations"):
                input_act = activation_importance.input_activations.view(-1)
                input_act = input_act.to(device=param.device, dtype=abs_param.dtype)

                if input_act.numel() >= cols:
                    # 官方 SSU/Wanda 风格：
                    # weight magnitude × input activation importance
                    scores = col_magnitudes * input_act[:cols]
                else:
                    print(
                        f"[LA-SSU] Warning: activation dim mismatch. "
                        f"act={input_act.numel()}, cols={cols}. "
                        f"Fallback to weight magnitude."
                    )

            return scores.float().cpu(), "column"

        # 1D parameters: each element is treated as one column-like unit
        elif param.dim() == 1:
            num_units = param.numel()
            scores = abs_param.view(-1)

            if activation_importance is not None and hasattr(activation_importance, "input_activations"):
                input_act = activation_importance.input_activations.view(-1)
                input_act = input_act.to(device=param.device, dtype=abs_param.dtype)

                if input_act.numel() >= num_units:
                    scores = scores * input_act[:num_units]
                else:
                    # 1D 参数通常很小，fallback 问题不大
                    pass

            return scores.float().cpu(), "element"

        else:
            return None, None


def collect_lassu_column_scores(
    model,
    calibration_data,
    num_calibration_samples=128,
    top_ratio=0.5,
    save_path=None,
    skip_embeddings_and_head=True,
):
    """
    LA-SSU 第一步：
        对一种语言的 calibration data 计算所有参数的 column-wise SSU/Wanda 分数。

    重要：
        “每种语言分别计算”不是修改分数公式。
        它只是用不同语言的 calibration data 分别跑同一套官方 SSU/Wanda 计算流程。

    例如：
        ibo calibration data -> ibo important columns
        hau calibration data -> hau important columns
        kir calibration data -> kir important columns

    参数：
        model:
            当前模型。

        calibration_data:
            某一种语言的 calibration dataloader。
            例如 ibo/hau/kir/npi/amh 各自的 calibration 数据。

        num_calibration_samples:
            用多少 calibration samples。
            先用 128 即可。

        top_ratio:
            每个参数里取 top 多少比例的 column 作为该语言的重要 column。
            如果想和官方 SSU 50% freezing ratio 对齐，先用 0.5。

        save_path:
            保存 score 文件路径。
            例如：
                /root/autodl-fs/ssu_scores/lassu/ibo_scores.pt

        skip_embeddings_and_head:
            默认跳过 embedding 和 lm_head。
            这和 SSU 论文以及你现有代码逻辑一致。

    返回：
        score_dict
    """
    if top_ratio <= 0.0 or top_ratio > 1.0:
        raise ValueError(f"[LA-SSU] top_ratio should be in (0, 1], got {top_ratio}")

    actual_model = _lassu_get_actual_model(model)

    print("=" * 80)
    print("[LA-SSU] Collecting column-wise SSU/Wanda scores")
    print(f"[LA-SSU] num_calibration_samples = {num_calibration_samples}")
    print(f"[LA-SSU] top_ratio = {top_ratio}")
    print(f"[LA-SSU] skip_embeddings_and_head = {skip_embeddings_and_head}")
    print("=" * 80)

    # 复用官方 SSU 的 calibration activation 收集函数。
    activation_stats = None
    if calibration_data is not None:
        activation_stats = _collect_activation_statistics(
            model,
            calibration_data,
            num_calibration_samples,
        )
    else:
        print("[LA-SSU] Warning: calibration_data is None. Fallback to magnitude-only scores.")

    score_dict = {}

    total_units = 0
    total_top_units = 0
    processed_params = 0

    for module_name, module in actual_model.named_modules():
        if module_name == "":
            continue

        if not list(module.parameters(recurse=False)):
            continue

        if _should_skip_module(module_name, skip_embeddings_and_head):
            print(f"[LA-SSU] Skipping module: {module_name}")
            continue

        for param_name, param in [
            ("weight", getattr(module, "weight", None)),
            ("bias", getattr(module, "bias", None)),
        ]:
            if param is None:
                continue

            if not isinstance(param, torch.Tensor):
                continue

            if param.requires_grad is False:
                continue

            if param.dim() not in [1, 2]:
                continue

            full_param_name = f"{module_name}.{param_name}"
            activation_importance = None

            if activation_stats is not None:
                activation_importance = activation_stats.get(full_param_name, None)

            scores, unit_axis = _lassu_compute_unit_scores(
                param,
                activation_importance=activation_importance,
            )

            if scores is None:
                continue

            num_units = int(scores.numel())
            if num_units == 0:
                continue

            top_k = max(1, int(num_units * top_ratio))
            top_idx = torch.topk(scores, top_k, largest=True).indices.cpu()

            score_dict[full_param_name] = {
                "scores": scores,
                "top_idx": top_idx,
                "num_units": num_units,
                "top_ratio": float(top_ratio),
                "unit_axis": unit_axis,
                "param_shape": tuple(param.shape),
                "score_type": "ssu_wanda_column",
            }

            processed_params += 1
            total_units += num_units
            total_top_units += top_k

            print(
                f"[LA-SSU] {full_param_name}: "
                f"axis={unit_axis}, selected top {top_k}/{num_units} units "
                f"({top_k / max(1, num_units):.2%})"
            )

    score_dict["__meta__"] = {
        "method": "LA-SSU",
        "score_type": "ssu_wanda_column",
        "top_ratio": float(top_ratio),
        "num_calibration_samples": int(num_calibration_samples),
        "skip_embeddings_and_head": bool(skip_embeddings_and_head),
        "processed_params": int(processed_params),
        "total_units": int(total_units),
        "total_top_units": int(total_top_units),
    }

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(score_dict, save_path)
        print(f"[LA-SSU] Saved column scores to: {save_path}")

    print("=" * 80)
    print("[LA-SSU] Score collection completed")
    print(f"[LA-SSU] processed params: {processed_params}")
    print(f"[LA-SSU] total units: {total_units:,}")
    print(f"[LA-SSU] selected top units: {total_top_units:,}")
    print("=" * 80)

    return score_dict


def build_lassu_column_scales(
    old_score_paths,
    current_score_path,
    old_shared_threshold=2,
    old_specific_scale=0.0,
    old_shared_scale=0.1,
    current_shared_scale=0.5,
    current_specific_scale=1.0,
    others_scale=1.0,
):
    """
    LA-SSU 第二步：
        根据历史语言和当前语言的重要 column overlap，生成每个 column 的 update scale。

    输入：
        old_score_paths:
            历史语言 score 文件列表。
            例如训练 kir 时：
                [ibo_scores.pt, hau_scores.pt]

        current_score_path:
            当前语言 score 文件。
            例如训练 kir 时：
                kir_scores.pt

    分类：
        old_union:
            任意历史语言重要的 columns。

        old_shared:
            至少被 old_shared_threshold 个历史语言认为重要的 columns。

        current_top:
            当前语言重要 columns。

    规则：
        old_union - current_top:
            旧语言重要、当前语言不重要。
            默认 scale=0.0，强保护。

        old_shared - current_top:
            多个旧语言共享、当前语言不重要。
            默认 scale=0.1，小幅更新。

        current_top ∩ old_union:
            当前语言和历史语言都重要。
            默认 scale=0.5，中等更新。

        current_top - old_union:
            当前语言特异重要。
            默认 scale=1.0，正常更新。

        others:
            默认 scale=1.0。
    """
    if isinstance(old_score_paths, str):
        old_score_paths = [p for p in old_score_paths.split(",") if p.strip()]

    old_score_paths = old_score_paths or []

    print("=" * 80)
    print("[LA-SSU] Building language-aware column scales")
    print(f"[LA-SSU] old_score_paths = {old_score_paths}")
    print(f"[LA-SSU] current_score_path = {current_score_path}")
    print(f"[LA-SSU] old_shared_threshold = {old_shared_threshold}")
    print(
        "[LA-SSU] scales: "
        f"old_specific={old_specific_scale}, "
        f"old_shared={old_shared_scale}, "
        f"current_shared={current_shared_scale}, "
        f"current_specific={current_specific_scale}, "
        f"others={others_scale}"
    )
    print("=" * 80)

    old_scores_list = [_lassu_score_file_load(p) for p in old_score_paths]
    current_scores = _lassu_score_file_load(current_score_path)

    scale_dict = {}

    for full_param_name, cur_info in current_scores.items():
        if full_param_name == "__meta__":
            continue

        if not isinstance(cur_info, dict):
            continue

        if "top_idx" not in cur_info or "num_units" not in cur_info:
            continue

        num_units = int(cur_info["num_units"])
        unit_axis = cur_info.get("unit_axis", "column")
        current_top = set(cur_info["top_idx"].view(-1).long().tolist())

        # old_count[j] 表示第 j 个 column 被多少个历史语言选为重要。
        old_count = torch.zeros(num_units, dtype=torch.long)

        for old_scores in old_scores_list:
            if full_param_name not in old_scores:
                continue

            old_info = old_scores[full_param_name]

            if not isinstance(old_info, dict) or "top_idx" not in old_info:
                continue

            old_idx = old_info["top_idx"].view(-1).long()
            old_idx = old_idx[(old_idx >= 0) & (old_idx < num_units)]
            old_count[old_idx] += 1

        old_union = set(torch.nonzero(old_count > 0).view(-1).tolist())
        old_shared = set(torch.nonzero(old_count >= old_shared_threshold).view(-1).tolist())

        # 默认正常更新。
        scale = torch.full((num_units,), float(others_scale), dtype=torch.float32)

        # 1. 旧语言重要：先强保护。
        for j in old_union:
            scale[j] = float(old_specific_scale)

        # 2. 多个旧语言共享：允许小幅更新。
        for j in old_shared:
            scale[j] = float(old_shared_scale)

        # 3. 当前语言重要：优先级最高。
        #    如果当前语言也需要这个 column，就不能完全冻结。
        for j in current_top:
            if j in old_union:
                scale[j] = float(current_shared_scale)
            else:
                scale[j] = float(current_specific_scale)

        scale_dict[full_param_name] = {
            "scale": scale,
            "unit_axis": unit_axis,
            "num_units": num_units,
            "param_shape": cur_info.get("param_shape", None),
        }

        overlap = current_top & old_union

        print(
            f"[LA-SSU] {full_param_name}: "
            f"axis={unit_axis}, "
            f"old_union={len(old_union)}, "
            f"old_shared={len(old_shared)}, "
            f"current_top={len(current_top)}, "
            f"current_old_overlap={len(overlap)}"
        )

    print(f"[LA-SSU] Built scale tensors for {len(scale_dict)} parameters")
    return scale_dict


def apply_lassu_column_scales(model, scale_dict):
    """
    LA-SSU 第三步：
        把 build_lassu_column_scales 生成的 scale 应用到模型参数上。

    对 2D weight：
        scale[j] 作用到第 j 列：
            weight[:, j] *= scale[j] in gradient space

    对 1D 参数：
        scale[j] 作用到第 j 个元素。

    注意：
        这里不是直接修改 weight 数值。
        而是在 backward 时修改梯度：
            grad = grad * scale_mask
    """
    actual_model = _lassu_get_actual_model(model)
    param_map = dict(actual_model.named_parameters())

    total_scaled_params = 0
    params_applied = 0

    print("=" * 80)
    print("[LA-SSU] Applying column-wise soft update scales")
    print("=" * 80)

    for full_param_name, info in scale_dict.items():
        scale = info["scale"]
        unit_axis = info.get("unit_axis", "column")

        resolved_name, param = _lassu_lookup_param(param_map, full_param_name)

        if param is None:
            print(f"[LA-SSU] Warning: cannot find parameter {full_param_name}, skip.")
            continue

        if not param.requires_grad:
            print(f"[LA-SSU] Warning: {resolved_name} requires_grad=False, skip.")
            continue

        scale = scale.to(device=param.device, dtype=param.dtype)

        if param.dim() == 2 and unit_axis == "column":
            rows, cols = param.shape

            if scale.numel() != cols:
                print(
                    f"[LA-SSU] Warning: scale dim mismatch for {resolved_name}: "
                    f"scale={scale.numel()}, cols={cols}. Skip."
                )
                continue

            scale_mask = scale.view(1, cols).expand(rows, cols)

        elif param.dim() == 1 and unit_axis == "element":
            if scale.numel() != param.numel():
                print(
                    f"[LA-SSU] Warning: scale dim mismatch for {resolved_name}: "
                    f"scale={scale.numel()}, numel={param.numel()}. Skip."
                )
                continue

            scale_mask = scale.view_as(param)

        else:
            print(
                f"[LA-SSU] Warning: unsupported param/unit for {resolved_name}: "
                f"param_dim={param.dim()}, unit_axis={unit_axis}. Skip."
            )
            continue

        param.register_hook(_make_scale_hook(scale_mask))

        with torch.no_grad():
            unique_values, counts = torch.unique(scale.float().cpu(), return_counts=True)
            scale_summary = ", ".join(
                [f"{float(v):.3f}:{int(c)}" for v, c in zip(unique_values, counts)]
            )

        total_scaled_params += int(param.numel())
        params_applied += 1

        print(
            f"[LA-SSU] Applied scale to {resolved_name}, "
            f"shape={tuple(param.shape)}, "
            f"unit_axis={unit_axis}, "
            f"scale_summary=({scale_summary})"
        )

    actual_model._lassu_scaled_params = total_scaled_params
    actual_model._lassu_params_applied = params_applied
    actual_model._hft_freeze_strategy = "la_ssu_column_soft_scaling"

    print("=" * 80)
    print("[LA-SSU] Soft scale hooks installed")
    print(f"[LA-SSU] params applied: {params_applied}")
    print(f"[LA-SSU] total parameters covered by scale hooks: {total_scaled_params:,}")
    print("=" * 80)

    return {
        "params_applied": params_applied,
        "total_scaled_params": total_scaled_params,
    }


def apply_lassu_from_score_files(
    model,
    old_score_paths,
    current_score_path,
    old_shared_threshold=2,
    old_specific_scale=0.0,
    old_shared_scale=0.1,
    current_shared_scale=0.5,
    current_specific_scale=1.0,
    others_scale=1.0,
):
    """
    LA-SSU 训练阶段总入口。

    后续在 main.py 里只需要调用这个函数。

    示例：
        apply_lassu_from_score_files(
            model=model,
            old_score_paths=[
                "/root/autodl-fs/ssu_scores/lassu/ibo_scores.pt",
                "/root/autodl-fs/ssu_scores/lassu/hau_scores.pt",
            ],
            current_score_path="/root/autodl-fs/ssu_scores/lassu/kir_scores.pt",
            old_specific_scale=0.0,
            old_shared_scale=0.1,
            current_shared_scale=0.5,
            current_specific_scale=1.0,
            others_scale=1.0,
        )
    """
    scale_dict = build_lassu_column_scales(
        old_score_paths=old_score_paths,
        current_score_path=current_score_path,
        old_shared_threshold=old_shared_threshold,
        old_specific_scale=old_specific_scale,
        old_shared_scale=old_shared_scale,
        current_shared_scale=current_shared_scale,
        current_specific_scale=current_specific_scale,
        others_scale=others_scale,
    )

    return apply_lassu_column_scales(model, scale_dict)



def _freeze_ssu_based_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, calibration_data=None, num_calibration_samples=128, axis_preference: Optional[str] = None):
    """
    Freeze parameters based on the SSU method. Parameters with higher
    SSU (i.e., Wanda) scores are frozen to avoid catastrophic forgetting, as they are more important
    for the model's current knowledge and capabilities.

    SSU score combines weight magnitude with activation importance computed from calibration data.
    For freezing (rather than pruning), we freeze parameters with high SSU scores to preserve
    important knowledge while allowing parameters with low scores to adapt during training.
    
    Args:
        model: The model to apply SSU-based freezing to
        freeze_ratio: Ratio of parameters to freeze (0.0 to 1.0) - freezes highest SSU score params
        seed: Random seed for reproducibility
        skip_embeddings_and_head: Whether to skip freezing embedding layers and lm_head
        calibration_data: DataLoader or iterable with calibration samples for computing activation importance
        num_calibration_samples: Number of samples to use for calibration (default: 128)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    print(f"Applying SSU-based parameter freezing with ratio: {freeze_ratio}{skip_msg}")
    print("Strategy: Freeze high SSU score weights (preserve knowledge), train low SSU score weights (allow adaptation)")
    print("SSU (Wanda) Score = |weight| * input_activation_importance (from preceding layer outputs)")
    
    # Collect activation statistics if calibration data is provided
    activation_stats = None
    if calibration_data is not None:
        print(f"Computing input activation statistics from {num_calibration_samples} calibration samples...")
        activation_stats = _collect_activation_statistics(model, calibration_data, num_calibration_samples)
        print("Input activation statistics collection completed.")
    else:
        print("Note: No calibration data provided, using weight magnitude as proxy for activation importance")
    
    total_params = 0
    frozen_params = 0
    skipped_params = 0
    modules_processed = 0
    skipped_modules = 0
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Iterate through all named modules
    for name, module in actual_model.named_modules():
        # Skip the root module
        if name == "":
            continue
            
        # Get all parameters in this module (non-recursive to avoid double-counting)
        module_params = list(module.parameters(recurse=False))
        if not module_params:
            continue
        
        # Check if this module should be skipped
        if _should_skip_module(name, skip_embeddings_and_head):
            module_param_count = sum(p.numel() for p in module_params)
            skipped_params += module_param_count
            total_params += module_param_count
            skipped_modules += 1
            print(f"\nSkipping module '{name}' ({type(module).__name__}): {module_param_count:,} parameters")
            continue
            
        modules_processed += 1
        module_total_params = 0
        module_frozen_params = 0
        
        print(f"\nProcessing module '{name}' ({type(module).__name__})")
        
        for param_idx, param in enumerate(module_params):
            param_name = f"param_{param_idx}"
            if hasattr(module, 'weight') and param is module.weight:
                param_name = "weight"
            elif hasattr(module, 'bias') and param is module.bias:
                param_name = "bias"
            
            # Skip frozen parameters
            if param.requires_grad is False:
                print(f"  Skipping frozen parameter '{param_name}' in module '{name}'")
                continue
            
            param_size = param.numel()
            module_total_params += param_size
            total_params += param_size

            # Apply SSU-based freezing
            frozen_count = _freeze_by_ssu_score(param, freeze_ratio, param_name, name, activation_stats, axis_preference)
            module_frozen_params += frozen_count
            frozen_params += frozen_count
            
            if frozen_count > 0:
                frozen_percentage = frozen_count / param_size
                print(f"  {param_name} ({list(param.shape)}): {frozen_count}/{param_size} elements frozen "
                      f"({frozen_percentage:.2%}) - high Wanda score weights")
        
        if module_total_params > 0:
            print(f"  Module total: {module_frozen_params}/{module_total_params} parameters frozen "
                  f"({module_frozen_params/module_total_params:.2%})")
    
    # Account for lm_head as it is not part of the main model
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        if lm_head_params > 0:
            skipped_params += lm_head_params
            total_params += lm_head_params
            skipped_modules += 1
            print(f"\nSkipping lm_head module: {lm_head_params:,} parameters (not frozen)")
    
    # Store the frozen parameter count in the model for accurate counting
    actual_model._hft_frozen_params = frozen_params
    actual_model._hft_total_params = total_params
    actual_model._hft_freeze_strategy = "ssu"

    print(f"\nSSU-based freezing completed:")
    print(f"  - Processed {modules_processed} modules")
    if skipped_modules > 0:
        print(f"  - Skipped {skipped_modules} modules (embeddings/lm_head): {skipped_params:,} parameters")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")
    print(f"  - Strategy: High SSU score weights frozen (preserve knowledge)")
    print(f"  - Strategy: Low SSU score weights trainable (allow adaptation)")


def _freeze_by_ssu_score(param, freeze_ratio, param_name, module_name, activation_stats=None, axis_preference: Optional[str] = None):
    """
    Freeze parameters based on their SSU scores. SSU scores combine weight magnitude
    with input activation importance computed from calibration data.

    The SSU score is computed as: |weight| * input_activation_importance
    Where input_activation_importance comes from the outputs of the immediately preceding layer.
    
    Args:
        param: The parameter tensor to apply SSU-based freezing to
        freeze_ratio: Ratio of parameters to freeze (highest SSU score ones)
        param_name: Name of the parameter (for logging)
        module_name: Name of the module (for logging)
        activation_stats: Dictionary containing input activation statistics from calibration data
    
    Returns:
        int: Number of elements frozen
    """
    if param.numel() == 0:
        return 0
    
    # Compute SSU scores based on weight magnitudes and input activation importance
    abs_param = torch.abs(param)
    original_shape = param.shape
    
    # Get input activation importance for this parameter
    activation_importance = None
    if activation_stats is not None:
        param_key = f"{module_name}.{param_name}"
        activation_importance = activation_stats.get(param_key, None)
    
    # For structured parameters (like linear layer weights), apply structured SSU-based freezing
    if len(original_shape) == 2 and "weight" in param_name:
        return _freeze_structured_ssu_2d(param, freeze_ratio, original_shape, activation_importance, axis_preference)
    else:
        # For other parameters, use element-wise SSU-based freezing
        return _freeze_unstructured_ssu(param, freeze_ratio, original_shape, activation_importance)


def _freeze_structured_ssu_2d(param, freeze_ratio, shape, activation_importance=None, axis_preference: Optional[str] = None):
    """
    Freeze entire rows or columns in a 2D weight matrix based on their SSU scores.
    This preserves the structured nature of linear transformations while using SSU
    scoring to identify important neurons/features.

    SSU uses input activations (outputs from the immediately preceding layer) to
    compute importance scores for the input features of the current layer.
    
    Args:
        param: The 2D parameter tensor
        freeze_ratio: Ratio of rows/columns to freeze
        shape: Shape of the parameter tensor
        activation_importance: Input activation importance statistics from calibration data
    """
    rows, cols = shape
    abs_param = torch.abs(param)
    
    # Compute Wanda scores using input activation importance (Wanda methodology)
    if activation_importance is not None:
        # Use actual input activation statistics if available
        if hasattr(activation_importance, 'input_activations'):
            # For linear layers: weight shape is [out_features, in_features]
            # Input activations correspond to the input features (columns)
            input_act_importance = activation_importance.input_activations.view(-1)
            # Ensure device/dtype match parameter
            input_act_importance = input_act_importance.to(device=param.device, dtype=abs_param.dtype)
            
            # Ensure the input activation size matches the input dimension (columns)
            if input_act_importance.numel() >= cols:
                # Column Wanda scores: weight magnitude * input activation importance
                col_magnitudes = torch.norm(abs_param, dim=0)  # [cols]
                col_wanda_scores = col_magnitudes * input_act_importance[:cols]
                
                # For output features (rows), we can use weight magnitude only
                # since Wanda primarily focuses on input activations
                row_magnitudes = torch.norm(abs_param, dim=1)  # [rows]
                row_wanda_scores = row_magnitudes
                
                # Prefer column-wise freezing when we have input activation data
                use_columns = True
            else:
                # Fallback if size mismatch
                use_columns = False
        else:
            # Fallback to magnitude-based proxy if activation format is unexpected
            use_columns = False
    else:
        use_columns = False
    
    # Compute scores based on whether we use input activations or fallback
    if not use_columns:
        # Fallback: Use weight-based proxy for activation importance
        row_magnitudes = torch.norm(abs_param, dim=1)  # [rows] - output neuron magnitudes
        row_variances = torch.var(abs_param, dim=1)    # [rows] - diversity of connections
        row_wanda_scores = row_magnitudes * (1.0 + row_variances)  # Higher variance = more important
        
        # Column Wanda score: combines input feature importance with output connectivity  
        col_magnitudes = torch.norm(abs_param, dim=0)  # [cols] - input feature magnitudes
        col_variances = torch.var(abs_param, dim=0)    # [cols] - diversity of connections
        col_wanda_scores = col_magnitudes * (1.0 + col_variances)  # Higher variance = more important
        
        # Decide whether to freeze by rows or columns based on score variance
        row_wanda_std = torch.std(row_wanda_scores)
        col_wanda_std = torch.std(col_wanda_scores)
        use_columns = col_wanda_std >= row_wanda_std
    
    # Decide whether to freeze by rows (output neurons) or columns (input features)
    # When using true Wanda (with input activations), prefer column-wise freezing
    row_wanda_std = torch.std(row_wanda_scores)
    col_wanda_std = torch.std(col_wanda_scores)

    # Respect explicit axis preference when provided
    if axis_preference == 'row':
        use_columns = False
    elif axis_preference == 'column':
        use_columns = True
    else:
        use_columns = True # Default to column-wise

    if use_columns:  # Prefer columns when we have input activation data or requested
        num_to_freeze = int(cols * freeze_ratio)
        if num_to_freeze > 0:
            # Get indices of highest Wanda score columns (input features)
            _, highest_indices = torch.topk(col_wanda_scores, num_to_freeze, largest=True)
            
            frozen_mask = torch.zeros_like(param, dtype=torch.bool)
            frozen_mask[:, highest_indices] = True
            
            # Store the mask in the parameter for gradient computation
            param.register_hook(_make_mask_hook(frozen_mask))
            
            avg_frozen_wanda = col_wanda_scores[highest_indices].mean().item()
            avg_frozen_magnitude = torch.norm(abs_param[:, highest_indices], dim=0).mean().item()
            print(f"    Froze {num_to_freeze} columns (input features) with avg Wanda score: {avg_frozen_wanda:.6f}, avg magnitude: {avg_frozen_magnitude:.6f}")
            
            return num_to_freeze * rows
    else:  # Use rows when input activations unavailable or for fallback / requested
        num_to_freeze = int(rows * freeze_ratio)
        if num_to_freeze > 0:
            # Get indices of highest Wanda score rows (output neurons)
            _, highest_indices = torch.topk(row_wanda_scores, num_to_freeze, largest=True)
            
            frozen_mask = torch.zeros_like(param, dtype=torch.bool)
            frozen_mask[highest_indices, :] = True
            
            # Store the mask in the parameter for gradient computation
            param.register_hook(_make_mask_hook(frozen_mask))
            
            avg_frozen_wanda = row_wanda_scores[highest_indices].mean().item()
            avg_frozen_magnitude = torch.norm(abs_param[highest_indices, :], dim=1).mean().item()
            print(f"    Froze {num_to_freeze} rows (output neurons) with avg Wanda score: {avg_frozen_wanda:.6f}, avg magnitude: {avg_frozen_magnitude:.6f}")
            
            return num_to_freeze * cols
    
    return 0


def _freeze_unstructured_ssu(param, freeze_ratio, original_shape, activation_importance=None):
    """
    Freeze individual elements based on their SSU scores.

    Args:
        param: The parameter tensor
        freeze_ratio: Ratio of elements to freeze
        original_shape: Original shape of the parameter
        activation_importance: Input activation importance statistics from calibration data
    """
    abs_param = torch.abs(param)
    flat_param = abs_param.view(-1)
    num_elements = flat_param.numel()
    num_to_freeze = int(num_elements * freeze_ratio)
    
    if num_to_freeze > 0:
        # Compute activation importance scores using input activations (SSU methodology)
        if activation_importance is not None and hasattr(activation_importance, 'element_importance'):
            # Use actual input activation statistics if available
            activation_scores = activation_importance.element_importance.view(-1)[:num_elements]
            activation_scores = activation_scores.to(device=param.device, dtype=flat_param.dtype)
            if activation_scores.numel() != num_elements:
                # Resize if dimensions don't match
                activation_scores = torch.nn.functional.interpolate(
                    activation_scores.unsqueeze(0).unsqueeze(0), 
                    size=num_elements, 
                    mode='linear', 
                    align_corners=False
                ).squeeze()
        else:
            # Fallback: Compute local importance scores as a proxy for input activation importance
            # Use local variance to approximate how much each weight affects the local computation
            if num_elements > 1:
                # For 1D parameters (like bias), use neighboring variance
                if len(original_shape) == 1:
                    # Create a simple local variance metric for 1D tensors
                    padded_param = torch.nn.functional.pad(flat_param.unsqueeze(0), (1, 1), mode='reflect').squeeze(0)
                    activation_scores = torch.zeros_like(flat_param)
                    for i in range(num_elements):
                        neighborhood = padded_param[i:i+3]  # 3-element neighborhood
                        activation_scores[i] = torch.var(neighborhood)
                else:
                    # For multi-dimensional parameters, compute local variance differently
                    reshaped = abs_param.view(-1)
                    # Use a moving window approach for local variance
                    window_size = min(5, num_elements)
                    activation_scores = torch.zeros_like(reshaped)
                    for i in range(num_elements):
                        start = max(0, i - window_size // 2)
                        end = min(num_elements, i + window_size // 2 + 1)
                        activation_scores[i] = torch.var(reshaped[start:end])
            else:
                activation_scores = torch.ones_like(flat_param)
        # Ensure device/dtype alignment for fallback path as well
        if 'activation_scores' in locals():
            activation_scores = activation_scores.to(device=param.device, dtype=flat_param.dtype)
        
        # Compute Wanda scores: weight magnitude * input activation importance
        wanda_scores = flat_param * (1.0 + activation_scores)  # Add 1 to avoid zero scores
        
        # Get indices of highest Wanda score elements
        _, highest_indices = torch.topk(wanda_scores, num_to_freeze, largest=True)
        
        # Create mask in original shape
        frozen_mask = torch.zeros(num_elements, dtype=torch.bool, device=param.device)
        frozen_mask[highest_indices] = True
        frozen_mask = frozen_mask.view(original_shape)
        
        # Store the mask in the parameter for gradient computation
        param.register_hook(_make_mask_hook(frozen_mask))
        
        # Log statistics
        frozen_wanda_scores = wanda_scores[highest_indices]
        frozen_magnitudes = flat_param[highest_indices]
        avg_frozen_wanda = frozen_wanda_scores.mean().item()
        avg_frozen_magnitude = frozen_magnitudes.mean().item()
        min_frozen_wanda = frozen_wanda_scores.min().item()
        max_frozen_wanda = frozen_wanda_scores.max().item()

        print(f"    Froze elements with SSU score range: [{min_frozen_wanda:.6f}, {max_frozen_wanda:.6f}], avg: {avg_frozen_wanda:.6f}")
        print(f"    Corresponding magnitude avg: {avg_frozen_magnitude:.6f}")
        
        return num_to_freeze
    
    return 0


########## SSU-based freezing by element-wise
def _freeze_ssu_elementwise_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, calibration_data=None, num_calibration_samples=128):
    """
    Freeze parameters based on element-wise SSU (Weight AND Activation) scores. This provides
    the finest-grained control by evaluating each individual weight element based on its SSU score.

    Unlike structured SSU freezing, this approach freezes individual elements regardless of their
    position, providing maximum granularity for preserving the most important weights.
    
    Args:
        model: The model to apply element-wise SSU-based freezing to
        freeze_ratio: Ratio of parameters to freeze (0.0 to 1.0) - freezes highest SSU score elements
        seed: Random seed for reproducibility (used for tie-breaking)
        skip_embeddings_and_head: Whether to skip freezing embedding layers and lm_head
        calibration_data: DataLoader or iterable with calibration samples for computing activation importance
        num_calibration_samples: Number of samples to use for calibration (default: 128)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    print(f"Applying element-wise SSU (Wanda)-based parameter freezing with ratio: {freeze_ratio}{skip_msg}")
    print("Strategy: Freeze individual high Wanda score elements (finest-grained control)")
    print("Wanda Score = |weight| * input_activation_importance (from preceding layer outputs)")
    print("Note: This provides maximum granularity but may break some structural coherence")
    
    # Collect activation statistics if calibration data is provided
    activation_stats = None
    if calibration_data is not None:
        print(f"Computing input activation statistics from {num_calibration_samples} calibration samples...")
        activation_stats = _collect_activation_statistics(model, calibration_data, num_calibration_samples)
        print("Input activation statistics collection completed.")
    else:
        print("Note: No calibration data provided, using weight magnitude as proxy for activation importance")
    
    total_params = 0
    frozen_params = 0
    skipped_params = 0
    modules_processed = 0
    skipped_modules = 0
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Iterate through all named modules
    for name, module in actual_model.named_modules():
        # Skip the root module
        if name == "":
            continue
            
        # Get all parameters in this module (non-recursive to avoid double-counting)
        module_params = list(module.parameters(recurse=False))
        if not module_params:
            continue
        
        # Check if this module should be skipped
        if _should_skip_module(name, skip_embeddings_and_head):
            module_param_count = sum(p.numel() for p in module_params)
            skipped_params += module_param_count
            total_params += module_param_count
            skipped_modules += 1
            print(f"\nSkipping module '{name}' ({type(module).__name__}): {module_param_count:,} parameters")
            continue
            
        modules_processed += 1
        module_total_params = 0
        module_frozen_params = 0
        
        print(f"\nProcessing module '{name}' ({type(module).__name__})")
        
        for param_idx, param in enumerate(module_params):
            param_name = f"param_{param_idx}"
            if hasattr(module, 'weight') and param is module.weight:
                param_name = "weight"
            elif hasattr(module, 'bias') and param is module.bias:
                param_name = "bias"
            
            # Skip frozen parameters
            if param.requires_grad is False:
                print(f"  Skipping frozen parameter '{param_name}' in module '{name}'")
                continue
            
            param_size = param.numel()
            module_total_params += param_size
            total_params += param_size

            # Apply element-wise SSU-based freezing
            frozen_count = _freeze_elementwise_ssu(param, freeze_ratio, param_name, name, activation_stats)
            module_frozen_params += frozen_count
            frozen_params += frozen_count
            
            if frozen_count > 0:
                frozen_percentage = frozen_count / param_size
                print(f"  {param_name} ({list(param.shape)}): {frozen_count}/{param_size} elements frozen "
                      f"({frozen_percentage:.2%}) - highest individual Wanda scores")
        
        if module_total_params > 0:
            print(f"  Module total: {module_frozen_params}/{module_total_params} parameters frozen "
                  f"({module_frozen_params/module_total_params:.2%})")
    
    # Store the frozen parameter count in the model for accurate counting
    actual_model._hft_frozen_params = frozen_params
    actual_model._hft_total_params = total_params
    actual_model._hft_freeze_strategy = "ssu_elementwise"

    print(f"\nElement-wise SSU-based freezing completed:")
    print(f"  - Processed {modules_processed} modules")
    if skipped_modules > 0:
        print(f"  - Skipped {skipped_modules} modules (embeddings/lm_head): {skipped_params:,} parameters")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")
    print(f"  - Strategy: Highest individual SSU score elements frozen (finest-grained control)")
    print(f"  - Strategy: Individual elements with low SSU scores remain trainable")


def _freeze_elementwise_ssu(param, freeze_ratio, param_name, module_name, activation_stats=None):
    """
    Freeze individual parameter elements based on their SSU scores.
    This provides the finest-grained control by evaluating each weight element independently.
    
    Args:
        param: The parameter tensor to apply element-wise SSU-based freezing to
        freeze_ratio: Ratio of elements to freeze (highest SSU score ones)
        param_name: Name of the parameter (for logging)
        module_name: Name of the module (for logging)
        activation_stats: Dictionary containing input activation statistics from calibration data
    
    Returns:
        int: Number of elements frozen
    """
    if param.numel() == 0:
        return 0
    
    abs_param = torch.abs(param)
    original_shape = param.shape
    flat_param = abs_param.view(-1)
    num_elements = flat_param.numel()
    num_to_freeze = int(num_elements * freeze_ratio)
    
    if num_to_freeze == 0:
        return 0
    
    # Get input activation importance for this parameter
    activation_importance = None
    if activation_stats is not None:
        param_key = f"{module_name}.{param_name}"
        activation_importance = activation_stats.get(param_key, None)
    
    # Compute element-wise Wanda scores
    if activation_importance is not None and hasattr(activation_importance, 'input_activations'):
        # Use actual input activation statistics
        input_act_importance = activation_importance.input_activations.view(-1)
        input_act_importance = input_act_importance.to(device=param.device, dtype=flat_param.dtype)
        
        # Handle different parameter shapes
        if len(original_shape) == 2:  # Weight matrices
            # For 2D weights, broadcast input activations properly
            rows, cols = original_shape
            if input_act_importance.numel() >= cols:
                # Broadcast column-wise (input features) to all rows
                input_act_2d = input_act_importance[:cols].unsqueeze(0).expand(rows, -1)
                activation_scores = input_act_2d.reshape(-1)
            else:
                # Fallback to uniform importance if size mismatch
                activation_scores = torch.ones_like(flat_param)
        elif len(original_shape) == 1:  # Bias vectors
            # For 1D parameters, use input activations directly or broadcast
            if input_act_importance.numel() >= num_elements:
                activation_scores = input_act_importance[:num_elements]
            else:
                # Broadcast or repeat to match size
                if input_act_importance.numel() > 0:
                    repeat_factor = (num_elements + input_act_importance.numel() - 1) // input_act_importance.numel()
                    activation_scores = input_act_importance.repeat(repeat_factor)[:num_elements]
                else:
                    activation_scores = torch.ones_like(flat_param)
        else:
            raise ValueError(f"Unsupported parameter shape {original_shape} for element-wise Wanda freezing")
    else:
        # Fallback: Use local variance as proxy for activation importance
        activation_scores = _compute_local_importance_proxy(flat_param, original_shape)
    
    # Ensure activation_scores matches flat_param size
    if activation_scores.numel() != num_elements:
        if activation_scores.numel() > num_elements:
            activation_scores = activation_scores[:num_elements]
        else:
            # Pad with ones if too small
            padding = torch.ones(num_elements - activation_scores.numel(), device=param.device, dtype=flat_param.dtype)
            activation_scores = torch.cat([activation_scores, padding])

    # Align activation_scores device/dtype with parameter
    activation_scores = activation_scores.to(device=param.device, dtype=flat_param.dtype)
    
    # Compute element-wise Wanda scores: |weight| * activation_importance
    wanda_scores = flat_param * (1.0 + activation_scores)  # Add 1 to avoid zero scores
    
    # Get indices of highest Wanda score elements
    _, highest_indices = torch.topk(wanda_scores, num_to_freeze, largest=True)
    
    # Create mask in original shape
    frozen_mask = torch.zeros(num_elements, dtype=torch.bool, device=param.device)
    frozen_mask[highest_indices] = True
    frozen_mask = frozen_mask.view(original_shape)
    
    # Store the mask in the parameter for gradient computation
    param.register_hook(_make_mask_hook(frozen_mask))
    
    # Log statistics
    frozen_wanda_scores = wanda_scores[highest_indices]
    frozen_magnitudes = flat_param[highest_indices]
    frozen_activations = activation_scores[highest_indices]
    
    avg_frozen_wanda = frozen_wanda_scores.mean().item()
    avg_frozen_magnitude = frozen_magnitudes.mean().item()
    avg_frozen_activation = frozen_activations.mean().item()
    min_frozen_wanda = frozen_wanda_scores.min().item()
    max_frozen_wanda = frozen_wanda_scores.max().item()
    
    print(f"    Froze {num_to_freeze} elements with Wanda scores [{min_frozen_wanda:.6f}, {max_frozen_wanda:.6f}]")
    print(f"    Avg frozen - Wanda: {avg_frozen_wanda:.6f}, Magnitude: {avg_frozen_magnitude:.6f}, Activation: {avg_frozen_activation:.6f}")
    
    return num_to_freeze


def _compute_local_importance_proxy(flat_param, original_shape):
    """
    Compute local importance scores as a proxy for activation importance when calibration data is unavailable.
    """
    num_elements = flat_param.numel()
    
    if num_elements <= 1:
        return torch.ones_like(flat_param)
    
    # For 1D parameters (like bias), use neighboring variance
    if len(original_shape) == 1:
        # Create a simple local variance metric for 1D tensors
        padded_param = torch.nn.functional.pad(flat_param.unsqueeze(0), (1, 1), mode='reflect').squeeze(0)
        local_variance = torch.zeros_like(flat_param)
        for i in range(num_elements):
            neighborhood = padded_param[i:i+3]  # 3-element neighborhood
            local_variance[i] = torch.var(neighborhood) if neighborhood.numel() > 1 else 0.0
        return local_variance
    
    elif len(original_shape) == 2:
        # For 2D parameters, compute row and column variances
        rows, cols = original_shape
        param_2d = flat_param.view(rows, cols)
        
        # Row variance (for each output neuron)
        row_variances = torch.var(param_2d, dim=1, keepdim=True)  # [rows, 1]
        # Column variance (for each input feature)  
        col_variances = torch.var(param_2d, dim=0, keepdim=True)  # [1, cols]
        
        # Combine row and column variances
        combined_variance = row_variances + col_variances  # Broadcasting: [rows, cols]
        return combined_variance.view(-1)
    
    else:
        # For multi-dimensional parameters, use a moving window approach
        window_size = min(5, num_elements)
        local_variance = torch.zeros_like(flat_param)
        for i in range(num_elements):
            start = max(0, i - window_size // 2)
            end = min(num_elements, i + window_size // 2 + 1)
            neighborhood = flat_param[start:end]
            local_variance[i] = torch.var(neighborhood) if neighborhood.numel() > 1 else 0.0
        return local_variance
    


########## SSU-SparseGPT freezing

def _collect_sgpt_statistics(model, calibration_data, num_samples: int = 128):
    """
    Collect SparseGPT-style input statistics: per-layer E[x^2] (diagonal of input covariance) used
    to estimate importance for columns of weight matrices. This mirrors SparseGPT’s use of H=E[X^T X].
    We store only the diagonal to keep it lightweight.
    """
    actual_model = model.model if hasattr(model, 'model') else model
    device = next(actual_model.parameters()).device

    stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if not (isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor)):
                return
            x = inp[0].detach()
            if not x.is_floating_point():
                return
            # Flatten batch/time dims; last dim is features
            if x.dim() >= 2:
                # Bring features to last dim; assume already last as in linear
                x2 = (x ** 2).float()
                # Reduce over all but last
                reduce_dims = list(range(x2.dim() - 1))
                ex2 = x2.mean(dim=reduce_dims)  # shape: [features]
            else:
                ex2 = (x ** 2).float()
            key = f"{name}"
            if key not in stats:
                stats[key] = ex2.cpu()
            else:
                # Average accumulate
                prev = stats[key]
                # Align sizes if mismatch (rare); fallback to min length
                m = min(prev.numel(), ex2.numel())
                stats[key] = (prev[:m] + ex2[:m].cpu()) / 2.0
        return hook_fn

    # Register on modules with params (skip embeddings/lm_head by name heuristics)
    for name, module in actual_model.named_modules():
        if not list(module.parameters(recurse=False)):
            continue
        ln = name.lower()
        if 'embed' in ln or 'embedding' in ln or 'lm_head' in ln:
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Run a few samples
    seen = 0
    was_training = actual_model.training
    actual_model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            if seen >= num_samples:
                break
            from collections.abc import Mapping
            if isinstance(batch, Mapping):
                inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                bs = inputs.get('input_ids', next(iter(inputs.values())) if inputs else None)
                bs_val = bs.shape[0] if isinstance(bs, torch.Tensor) and bs.dim() > 0 else 1
                _ = actual_model(**inputs)
            elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], Mapping):
                inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch[0].items() if isinstance(v, torch.Tensor)}
                bs = inputs.get('input_ids', next(iter(inputs.values())) if inputs else None)
                bs_val = bs.shape[0] if isinstance(bs, torch.Tensor) and bs.dim() > 0 else 1
                _ = actual_model(**inputs)
            else:
                tens = batch[0] if isinstance(batch, (list, tuple)) else batch
                tens = tens.to(device) if hasattr(tens, 'to') else tens
                bs_val = tens.shape[0] if hasattr(tens, 'shape') and len(tens.shape) > 0 else 1
                _ = actual_model(input_ids=tens)
            seen += bs_val
            if seen >= num_samples:
                break

    for h in hooks:
        h.remove()
    if was_training:
        actual_model.train()

    # Package into simple object compatible with Wanda/Fisher element-wise interfaces where useful
    class SGPTImportance:
        def __init__(self, ex2: torch.Tensor):
            self.input_activations = ex2.view(-1)  # treat as input feature importance
            self.element_importance = self.input_activations

    processed = {}
    for module_name, ex2 in stats.items():
        processed[f"{module_name}.weight"] = SGPTImportance(ex2)
        processed[f"{module_name}.bias"] = SGPTImportance(ex2)

    print(f"Collected SparseGPT input statistics for {len(processed)} parameters from {seen} samples")
    return processed


def _freeze_sgpt_based_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, calibration_data=None, num_calibration_samples=128, axis_preference: Optional[str] = None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    print(f"Applying SparseGPT-based freezing with ratio: {freeze_ratio}")
    # Collect per-layer input E[x^2]
    sgpt_stats = None
    if calibration_data is not None:
        sgpt_stats = _collect_sgpt_statistics(model, calibration_data, num_calibration_samples)

    actual_model = model.model if hasattr(model, 'model') else model
    total_params = 0
    frozen_params = 0

    for name, module in actual_model.named_modules():
        if name == "":
            continue
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if _should_skip_module(name, skip_embeddings_and_head):
            total_params += sum(p.numel() for p in params)
            continue
        # Weight matrices: column scores ~ E[x^2] (importance of input features)
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor) and module.weight.dim() == 2:
            W = module.weight
            rows, cols = W.shape
            total_params += W.numel()
            key = f"{name}.weight"
            ex2 = sgpt_stats.get(key).input_activations.to(W.device, W.dtype) if (sgpt_stats is not None and key in sgpt_stats) else None
            if ex2 is None or ex2.numel() < cols:
                # fallback to magnitude-based proxy
                ex2 = torch.norm(W.detach().abs(), dim=0)
            else:
                ex2 = ex2[:cols]

            # Column-wise by default; row-wise if requested
            if axis_preference == 'row':
                # Use (W * ex2_cols) aggregated by rows as a simple proxy for row importance
                if sgpt_stats is not None and key in sgpt_stats and sgpt_stats[key].input_activations.numel() >= cols:
                    ex_cols = sgpt_stats[key].input_activations.to(W.device, W.dtype)[:cols]
                    row_scores = (W.detach().abs() * ex_cols.unsqueeze(0)).sum(dim=1)
                else:
                    row_scores = torch.norm(W.detach().abs(), dim=1)
                k = int(rows * freeze_ratio)
                if k > 0:
                    _, ridx = torch.topk(row_scores, k, largest=True)
                    mask = torch.zeros_like(W, dtype=torch.bool)
                    mask[ridx, :] = True
                    W.register_hook(_make_mask_hook(mask))
                    frozen_params += k * cols
                    print(f"  {name}.weight: froze {k} rows by SparseGPT row-scores")
            else:
                k = int(cols * freeze_ratio)
                if k > 0:
                    _, cidx = torch.topk(ex2, k, largest=True)
                    mask = torch.zeros_like(W, dtype=torch.bool)
                    mask[:, cidx] = True
                    W.register_hook(_make_mask_hook(mask))
                    frozen_params += k * rows
                    print(f"  {name}.weight: froze {k} columns by SparseGPT E[x^2]")

        # Bias and other params: element-wise by ex2 if available
        for attr in ["bias"]:
            if hasattr(module, attr):
                p = getattr(module, attr)
                if isinstance(p, torch.Tensor):
                    total_params += p.numel()
                    key = f"{name}.{attr}"
                    ex2 = sgpt_stats.get(key).element_importance.to(p.device, p.dtype) if (sgpt_stats is not None and key in sgpt_stats) else None
                    k = int(p.numel() * freeze_ratio)
                    if k > 0:
                        if ex2 is None or ex2.numel() < p.numel():
                            scores = p.detach().abs().view(-1)
                        else:
                            scores = ex2.view(-1)[:p.numel()]
                        _, idx = torch.topk(scores, k, largest=True)
                        mask = torch.zeros_like(p, dtype=torch.bool).view(-1)
                        mask[idx] = True
                        mask = mask.view_as(p)
                        p.register_hook(_make_mask_hook(mask))
                        frozen_params += k

    actual_model._hft_frozen_params = frozen_params
    if not hasattr(actual_model, '_hft_total_params'):
        actual_model._hft_total_params = sum(pp.numel() for pp in actual_model.parameters())
    actual_model._hft_freeze_strategy = "sparsegpt_based"
    print(f"SparseGPT-based freezing completed: {frozen_params} params masked")


def _freeze_sgpt_elementwise_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, calibration_data=None, num_calibration_samples=128):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    print(f"Applying SparseGPT element-wise freezing with ratio: {freeze_ratio}")
    sgpt_stats = None
    if calibration_data is not None:
        sgpt_stats = _collect_sgpt_statistics(model, calibration_data, num_calibration_samples)

    actual_model = model.model if hasattr(model, 'model') else model
    total_params = 0
    frozen_params = 0
    for name, module in actual_model.named_modules():
        if name == "":
            continue
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if _should_skip_module(name, skip_embeddings_and_head):
            total_params += sum(p.numel() for p in params)
            continue
        for attr in ["weight", "bias"]:
            if not hasattr(module, attr):
                continue
            p = getattr(module, attr)
            if not isinstance(p, torch.Tensor) or p.requires_grad is False:
                continue
            total_params += p.numel()
            key = f"{name}.{attr}"
            scores = None
            if sgpt_stats is not None and key in sgpt_stats:
                scores = sgpt_stats[key].element_importance.to(p.device, p.dtype).view(-1)
                if scores.numel() < p.numel():
                    scores = None
            if scores is None:
                scores = p.detach().abs().view(-1)
            k = int(p.numel() * freeze_ratio)
            if k <= 0:
                continue
            _, idx = torch.topk(scores, k, largest=True)
            mask = torch.zeros_like(p, dtype=torch.bool).view(-1)
            mask[idx] = True
            p.register_hook(_make_mask_hook(mask.view_as(p)))
            frozen_params += k

    actual_model._hft_frozen_params = frozen_params
    if not hasattr(actual_model, '_hft_total_params'):
        actual_model._hft_total_params = sum(pp.numel() for pp in actual_model.parameters())
    actual_model._hft_freeze_strategy = "sparsegpt_elementwise"
    print(f"SparseGPT element-wise freezing completed: {frozen_params}/{total_params} elements\n")



########## SSU-Fisher freezing


def _collect_fisher_information(model, calibration_data, num_samples: int = 128):
    """
    Collect diagonal Fisher Information estimates for each parameter using calibration data.

    We approximate the Fisher diagonal F_i = E[(\partial log p(y|x;\theta)/\partial \theta_i)^2]
    using the squared gradients of the negative log-likelihood loss per batch and average across samples.

    Args:
        model: HF model
        calibration_data: DataLoader yielding tokenized batches with labels
        num_samples: Max number of samples (examples) to use

    Returns:
        dict mapping "module_name.param" -> object with attributes:
            - element_importance: flattened fisher diag tensor
            - input_activations: alias to element_importance for reuse in Wanda-like code paths
    """
    actual_model = model.model if hasattr(model, 'model') else model
    device = next(actual_model.parameters()).device

    # Prepare accumulators lazily upon first gradient seen for a param
    fisher_accum = {}

    # We'll switch to eval mode for deterministic behavior (dropout off) but keep grads
    was_training = actual_model.training
    actual_model.eval()

    seen = 0
    try:
        for batch in calibration_data:
            if seen >= num_samples:
                break

            # Normalize inputs to device
            from collections.abc import Mapping
            if isinstance(batch, Mapping):
                inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], Mapping):
                inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch[0].items()}
            else:
                # Fallback: treat as input_ids only
                inputs = {'input_ids': batch.to(device) if hasattr(batch, 'to') else batch}

            batch_size = None
            if isinstance(inputs.get('input_ids', None), torch.Tensor):
                bs_t = inputs['input_ids']
                batch_size = bs_t.shape[0] if bs_t.dim() > 0 else 1
            else:
                batch_size = 1

            model.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
            if loss is None:
                # Try to compute loss if labels missing (rare with our collator)
                if 'labels' in inputs:
                    # labels supplied but no loss returned: compute manually
                    logits = outputs.logits
                    # Shift for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs['labels'][..., 1:].contiguous()
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    # Can't compute Fisher without loss
                    seen += batch_size
                    continue

            loss = loss.float()
            loss.backward()

            # Accumulate squared grads per parameter
            with torch.no_grad():
                for module_name, module in actual_model.named_modules():
                    params = list(module.parameters(recurse=False))
                    if not params:
                        continue
                    for param_attr, p in [("weight", getattr(module, 'weight', None)), ("bias", getattr(module, 'bias', None))]:
                        if p is None or p.grad is None:
                            continue
                        key = f"{module_name}.{param_attr}" if module_name else param_attr
                        g2 = (p.grad.detach() ** 2).to(dtype=torch.float32)
                        if key not in fisher_accum:
                            fisher_accum[key] = g2.clone()
                        else:
                            # Accumulate with shape match; if mismatch occurs skip to be safe
                            if fisher_accum[key].shape == g2.shape:
                                fisher_accum[key] += g2
                model.zero_grad(set_to_none=True)

            seen += batch_size
            if seen >= num_samples:
                break
    except Exception as e:
        print(f"Warning: Error during Fisher collection: {e}")
        fisher_accum = {}
    finally:
        # Restore training state
        if was_training:
            actual_model.train()

    if not fisher_accum:
        print("Fisher collection returned empty; will fallback to magnitude-based importance")
        return None

    # Normalize by number of samples to estimate expectation
    class FisherImportance:
        def __init__(self, elem_imp: torch.Tensor):
            # Store as 1D vector for element-wise; also provide Wanda-compatible attribute name
            self.element_importance = (elem_imp / max(1, seen)).contiguous().view(-1)
            self.input_activations = self.element_importance  # alias for reuse

    fisher_stats = {}
    for key, tensor in fisher_accum.items():
        fisher_stats[key] = FisherImportance(tensor)

    print(f"Collected Fisher information for {len(fisher_stats)} parameters from {seen} samples")

    # Proactively clear grads and cached buffers
    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    try:
        actual_model.zero_grad(set_to_none=True)
    except Exception:
        pass

    # Drop large temporary accumulators
    try:
        del fisher_accum
    except Exception:
        pass

    # Run garbage collection and empty device caches
    gc.collect()
    if hasattr(torch.cuda, "empty_cache") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
    return fisher_stats


def _freeze_fisher_based_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, calibration_data=None, num_calibration_samples=128, axis_preference: Optional[str] = None):
    """
    Freeze parameters based on Fisher information. Parameters with higher Fisher are frozen
    to preserve knowledge; lower-Fisher ones are left trainable to adapt.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    skip_msg = " (skipping embeddings and lm_head)" if skip_embeddings_and_head else ""
    print(f"Applying Fisher-based parameter freezing with ratio: {freeze_ratio}{skip_msg}")
    print("Strategy: Freeze high-Fisher weights (preserve knowledge), train low-Fisher weights (allow adaptation)")

    # Collect Fisher stats
    fisher_stats = None
    if calibration_data is not None:
        print(f"Computing Fisher information from {num_calibration_samples} calibration samples...")
        fisher_stats = _collect_fisher_information(model, calibration_data, num_calibration_samples)
        print("Fisher collection completed.")
    else:
        print("Note: No calibration data provided, falling back to magnitude-based proxy")

    total_params = 0
    frozen_params = 0
    skipped_params = 0
    modules_processed = 0
    skipped_modules = 0

    actual_model = model.model if hasattr(model, 'model') else model

    for name, module in actual_model.named_modules():
        if name == "":
            continue
        module_params = list(module.parameters(recurse=False))
        if not module_params:
            continue
        if _should_skip_module(name, skip_embeddings_and_head):
            module_param_count = sum(p.numel() for p in module_params)
            skipped_params += module_param_count
            total_params += module_param_count
            skipped_modules += 1
            print(f"\nSkipping module '{name}' ({type(module).__name__}): {module_param_count:,} parameters")
            continue

        modules_processed += 1
        module_total_params = 0
        module_frozen_params = 0

        print(f"\nProcessing module '{name}' ({type(module).__name__})")

        for param_idx, param in enumerate(module_params):
            param_name = f"param_{param_idx}"
            if hasattr(module, 'weight') and param is module.weight:
                param_name = "weight"
            elif hasattr(module, 'bias') and param is module.bias:
                param_name = "bias"

            if param.requires_grad is False:
                print(f"  Skipping frozen parameter '{param_name}' in module '{name}'")
                continue

            param_size = param.numel()
            module_total_params += param_size
            total_params += param_size

            frozen_count = _freeze_by_fisher_score(param, freeze_ratio, param_name, name, fisher_stats, axis_preference)
            module_frozen_params += frozen_count
            frozen_params += frozen_count

            if frozen_count > 0:
                frozen_percentage = frozen_count / param_size
                print(f"  {param_name} ({list(param.shape)}): {frozen_count}/{param_size} elements frozen ({frozen_percentage:.2%})")

        if module_total_params > 0:
            print(f"  Module total: {module_frozen_params}/{module_total_params} parameters frozen ({module_frozen_params/module_total_params:.2%})")

    # Account for lm_head outside model.model structures
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        if lm_head_params > 0:
            skipped_params += lm_head_params
            total_params += lm_head_params
            skipped_modules += 1
            print(f"\nSkipping lm_head module: {lm_head_params:,} parameters (not frozen)")

    actual_model._hft_frozen_params = frozen_params
    actual_model._hft_total_params = total_params
    actual_model._hft_freeze_strategy = "fisher_based"

    print(f"\nFisher-based freezing completed:")
    print(f"  - Processed {modules_processed} modules")
    if skipped_modules > 0:
        print(f"  - Skipped {skipped_modules} modules (embeddings/lm_head): {skipped_params:,} parameters")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    print(f"  - Trainable parameters: {total_params-frozen_params:,} ({(total_params-frozen_params)/total_params:.2%})")

    # Clean up Fisher stats and GPU caches to free memory
    try:
        if 'fisher_stats' in locals() and fisher_stats is not None:
            del fisher_stats
    except Exception:
        pass

    # Proactively clear gradients that may linger
    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        pass
    try:
        actual_model.zero_grad(set_to_none=True)
    except Exception:
        pass

    # Run garbage collection and empty device caches
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
    except Exception:
        pass


def _freeze_by_fisher_score(param, freeze_ratio, param_name, module_name, fisher_stats=None, axis_preference: Optional[str] = None):
    """
    Apply Fisher-based freezing to a parameter.
    For 2D weights, prefer structured (columns by default), else element-wise.
    """
    if param.numel() == 0:
        return 0

    original_shape = param.shape
    # Retrieve fisher element importance if available
    fisher_importance = None
    if fisher_stats is not None:
        param_key = f"{module_name}.{param_name}"
        fisher_importance = fisher_stats.get(param_key, None)

    if len(original_shape) == 2 and "weight" in param_name:
        return _freeze_structured_fisher_2d(param, freeze_ratio, original_shape, fisher_importance, axis_preference)
    else:
        return _freeze_unstructured_fisher(param, freeze_ratio, original_shape, fisher_importance)


def _freeze_structured_fisher_2d(param, freeze_ratio, shape, fisher_importance=None, axis_preference: Optional[str] = None):
    rows, cols = shape
    abs_param = torch.abs(param)

    # Build per-element fisher diag tensor (fallback to magnitude if not available)
    if fisher_importance is not None and hasattr(fisher_importance, 'element_importance'):
        elem = fisher_importance.element_importance.to(device=param.device, dtype=abs_param.dtype)
        if elem.numel() != rows * cols:
            # resize via interpolation fallback
            flat = elem.view(-1)
            if flat.numel() > 1:
                new = torch.nn.functional.interpolate(flat.unsqueeze(0).unsqueeze(0), size=rows*cols, mode='linear', align_corners=False).squeeze()
                elem = new
            else:
                elem = torch.ones(rows*cols, device=param.device, dtype=abs_param.dtype)
        fisher_2d = elem.view(rows, cols)
    else:
        # magnitude proxy
        fisher_2d = abs_param.detach()

    # Aggregate unit scores
    col_scores = fisher_2d.sum(dim=0)  # importance per input feature
    row_scores = fisher_2d.sum(dim=1)  # importance per output neuron

    # Decide axis
    use_columns = True
    if axis_preference == 'row':
        use_columns = False
    elif axis_preference == 'column':
        use_columns = True
    else:
        # choose the axis with higher separation
        use_columns = torch.std(col_scores) >= torch.std(row_scores)

    if use_columns:
        k = int(cols * freeze_ratio)
        if k <= 0:
            return 0
        _, idx = torch.topk(col_scores, k, largest=True)
        mask = torch.zeros_like(param, dtype=torch.bool)
        mask[:, idx] = True
        param.register_hook(_make_mask_hook(mask))
        print(f"    Froze {k} columns (input features) by Fisher scores")
        return k * rows
    else:
        k = int(rows * freeze_ratio)
        if k <= 0:
            return 0
        _, idx = torch.topk(row_scores, k, largest=True)
        mask = torch.zeros_like(param, dtype=torch.bool)
        mask[idx, :] = True
        param.register_hook(_make_mask_hook(mask))
        print(f"    Froze {k} rows (output neurons) by Fisher scores")
        return k * cols


def _freeze_unstructured_fisher(param, freeze_ratio, original_shape, fisher_importance=None):
    flat_num = param.numel()
    k = int(flat_num * freeze_ratio)
    if k <= 0:
        return 0
    if fisher_importance is not None and hasattr(fisher_importance, 'element_importance'):
        scores = fisher_importance.element_importance.to(device=param.device, dtype=param.dtype).view(-1)
        if scores.numel() != flat_num:
            if scores.numel() > 1:
                scores = torch.nn.functional.interpolate(scores.unsqueeze(0).unsqueeze(0), size=flat_num, mode='linear', align_corners=False).squeeze()
            else:
                scores = torch.ones(flat_num, device=param.device, dtype=param.dtype)
    else:
        # fallback to magnitude
        scores = param.detach().abs().view(-1)

    _, top_idx = torch.topk(scores, k, largest=True)
    mask = torch.zeros(flat_num, dtype=torch.bool, device=param.device)
    mask[top_idx] = True
    mask = mask.view(original_shape)
    param.register_hook(_make_mask_hook(mask))
    return k


def _freeze_fisher_elementwise_parameters(model, freeze_ratio=0.5, seed=None, skip_embeddings_and_head=False, calibration_data=None, num_calibration_samples=128):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    print(f"Applying element-wise Fisher-based freezing with ratio: {freeze_ratio}")
    fisher_stats = None
    if calibration_data is not None:
        fisher_stats = _collect_fisher_information(model, calibration_data, num_calibration_samples)

    total_params = 0
    frozen_params = 0
    actual_model = model.model if hasattr(model, 'model') else model
    for name, module in actual_model.named_modules():
        if name == "":
            continue
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if _should_skip_module(name, skip_embeddings_and_head):
            total_params += sum(p.numel() for p in params)
            continue
        for param_attr in ["weight", "bias"]:
            if not hasattr(module, param_attr):
                continue
            p = getattr(module, param_attr)
            if not isinstance(p, torch.Tensor) or p.requires_grad is False:
                continue
            total_params += p.numel()
            full_name = f"{name}.{param_attr}"
            fi = fisher_stats.get(full_name, None) if fisher_stats is not None else None
            k = _freeze_unstructured_fisher(p, freeze_ratio, p.shape, fi)
            frozen_params += k

    actual_model._hft_frozen_params = frozen_params
    actual_model._hft_total_params = total_params
    actual_model._hft_freeze_strategy = "fisher_elementwise"
    print(f"Element-wise Fisher freezing completed: {frozen_params}/{total_params} elements frozen ({(frozen_params/max(1,total_params)):.2%})")



##########


def _freeze_chat_template_tokens_topping(model, tokenizer, freeze_ratio=1.0, seed=None):
    """
    Freeze embeddings and output projections for chat template special tokens as a topping.
    
    This is applied in addition to other freezing strategies to preserve conversational
    structure while still allowing the main strategy to operate on other parameters.
    
    Args:
        model: The instruction-tuned model to freeze special tokens in
        tokenizer: Tokenizer to identify special token IDs
        freeze_ratio: Ratio of special tokens to freeze (0.0 to 1.0, default: 1.0 = all)
        seed: Random seed for reproducibility when freeze_ratio < 1.0
    
    Returns:
        None: Modifies model in-place by freezing specified token embeddings
    """
    if seed is not None:
        random.seed(seed + 1000)  # Offset seed to avoid conflicts with main strategy
        torch.manual_seed(seed + 1000)
    
    # Handle wrapped models
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Get all special token IDs from the tokenizer
    # TODO: Properly block special tokens in the tokenizer for OLMo, Qwen, and Phi
    # - <|endoftext|>, <|user|>, <|assistant|>, <|endofprompt|>, <|pad|> for OLMo
    # - <|im_start|>, user, <|im_end|>, assistant, <think>, </think> for Qwen
    # - <|endoftext|>, <|user|>, <|assistant|>, <|end|>, <|system|>, <|endofprompt|> for Phi (To be confirmed)
    special_token_ids = _get_special_token_ids(tokenizer)
    
    if not special_token_ids:
        print("Warning: No special tokens found in tokenizer. Nothing to freeze.")
        return
    
    # Apply freeze ratio to special tokens
    if freeze_ratio < 1.0:
        num_to_freeze = max(1, int(len(special_token_ids) * freeze_ratio))
        special_token_ids = random.sample(special_token_ids, num_to_freeze)
        print(f"Randomly selected {len(special_token_ids)} special tokens to freeze")
    
    # Count frozen parameters
    total_frozen_params = 0
    
    # Freeze input embeddings for special tokens
    embedding_frozen = _freeze_embedding_rows(actual_model, special_token_ids)
    total_frozen_params += embedding_frozen
    
    # Freeze output projection rows for special tokens
    output_frozen = _freeze_output_projection_rows(model, special_token_ids)
    total_frozen_params += output_frozen
    
    # Update metadata in model (add to existing counts if present)
    if hasattr(actual_model, '_hft_frozen_params'):
        actual_model._hft_frozen_params += total_frozen_params
        actual_model._hft_frozen_special_tokens = len(special_token_ids)
        actual_model._hft_freeze_strategy += " + chat_template_tokens"
    else:
        # If no previous strategy was applied
        total_params = sum(p.numel() for p in actual_model.parameters())
        actual_model._hft_frozen_params = total_frozen_params
        actual_model._hft_total_params = total_params
        actual_model._hft_freeze_strategy = "chat_template_tokens_only"
        actual_model._hft_frozen_special_tokens = len(special_token_ids)
    

def _get_special_token_ids(tokenizer):
    """
    Extract all special token IDs from the tokenizer.
    
    Args:
        tokenizer: The tokenizer to extract special tokens from
    
    Returns:
        list: List of special token IDs
    """
    special_token_ids = []

    def _add_token(tok_like):
        try:
            # tok_like can be str, AddedToken, or other object with .content
            t = getattr(tok_like, 'content', tok_like)
            if t is None:
                return
            token_id = tokenizer.convert_tokens_to_ids(t)
            if token_id is not None and token_id != tokenizer.unk_token_id and token_id != -1:
                special_token_ids.append(int(token_id))
        except Exception:
            return

    # Get special tokens from tokenizer attributes
    special_tokens_map = getattr(tokenizer, 'special_tokens_map', {}) or {}

    # Common special token attributes
    special_token_attrs = [
        'bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 
        'cls_token', 'mask_token', 'additional_special_tokens'
    ]
    
    # Collect special token IDs
    for attr in special_token_attrs:
        if hasattr(tokenizer, attr):
            token = getattr(tokenizer, attr)
            if token is not None:
                if isinstance(token, (list, tuple)):
                    # Handle additional_special_tokens which is a list
                    for t in token:
                        _add_token(t)
                else:
                    # Handle single tokens
                    _add_token(token)
    
    # Also get from special_tokens_map
    for _, token in special_tokens_map.items():
        if isinstance(token, (list, tuple)):
            for t in token:
                _add_token(t)
        else:
            _add_token(token)

    # Explicitly include common chat-template tokens for key model families when available
    olmo_tokens = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|endofprompt|>", "<|pad|>"]
    qwen_tokens = ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>", "<think>", "</think>"]
    phi_tokens  = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>", "<|system|>", "<|endofprompt|>"]

    # Heuristic selection: try to detect family; else try all sets
    name_hint = (getattr(tokenizer, 'name_or_path', '') or str(getattr(tokenizer, '__class__', type('T', (), {})).__name__)).lower()
    candidate_sets = []
    if any(k in name_hint for k in ["olmo", "allenai-olmo"]):
        candidate_sets.append(olmo_tokens)
    if any(k in name_hint for k in ["qwen"]):
        candidate_sets.append(qwen_tokens)
    if any(k in name_hint for k in ["phi"]):
        candidate_sets.append(phi_tokens)
    if not candidate_sets:
        candidate_sets = [olmo_tokens, qwen_tokens, phi_tokens]

    for token_list in candidate_sets:
        for t in token_list:
            _add_token(t)
    
    # Remove duplicates and sort
    special_token_ids = sorted(list(set(special_token_ids)))
    
    # Log the special tokens found
    print(f"Detected special tokens:")
    for token_id in special_token_ids:
        try:
            token_str = tokenizer.convert_ids_to_tokens(token_id)
        except Exception:
            token_str = "<unavailable>"
        print(f"  ID {token_id}: '{token_str}'")
    
    return special_token_ids


def _freeze_embedding_rows(model, token_ids):
    """
    Freeze specific rows in the input embedding matrix corresponding to given token IDs.
    
    Args:
        model: The model to freeze embeddings in
        token_ids: List of token IDs whose embeddings should be frozen
    
    Returns:
        int: Number of parameters frozen
    """
    frozen_params = 0
    
    # Find embedding layers
    embedding_layers = _find_embedding_layers(model)
    
    for name, embedding_layer in embedding_layers:
        if hasattr(embedding_layer, 'weight'):
            # Freeze specific rows in the embedding weight matrix
            param = embedding_layer.weight
            
            # Create a mask for the rows to freeze
            freeze_mask = torch.zeros_like(param, dtype=torch.bool)
            for token_id in token_ids:
                tid = int(token_id)
                if 0 <= tid < param.size(0):  # Ensure token_id is valid
                    freeze_mask[tid, :] = True # vocab size x embedding size 
            
            # Apply gradient masking hook
            def make_embedding_hook(mask):
                def hook_fn(grad):
                    if grad is not None:
                        return grad.masked_fill(mask.to(device=grad.device, dtype=torch.bool), 0.0)
                    return grad
                return hook_fn
            
            param.register_hook(make_embedding_hook(freeze_mask))
            
            # Count frozen parameters
            frozen_count = freeze_mask.sum().item()
            frozen_params += frozen_count
            
            print(f"  Froze {frozen_count} parameters in embedding layer '{name}' for {len(token_ids)} special tokens")
    
    return frozen_params


def _freeze_output_projection_rows(model, token_ids):
    """
    Freeze specific rows in the output projection matrix corresponding to given token IDs.
    
    Args:
        model: The model to freeze output projections in
        token_ids: List of token IDs whose output projections should be frozen
    
    Returns:
        int: Number of parameters frozen
    """
    frozen_params = 0
    
    # Find output projection layers (typically lm_head)
    output_layers = _find_output_projection_layers(model)
    
    max_id = max(token_ids) if token_ids else -1

    for name, output_layer in output_layers:
        if hasattr(output_layer, 'weight'):
            # Freeze specific rows or columns in the output projection weight matrix
            param = output_layer.weight

            if param.dim() != 2:
                continue

            rows, cols = param.size(0), param.size(1)

            # Prefer row-wise layout [vocab_size, hidden]
            if rows > max_id:
                freeze_mask = torch.zeros_like(param, dtype=torch.bool)
                for token_id in token_ids:
                    tid = int(token_id)
                    if 0 <= tid < rows:
                        freeze_mask[tid, :] = True

                if freeze_mask.any():
                    def make_output_hook(mask):
                        def hook_fn(grad):
                            if grad is not None:
                                return grad.masked_fill(mask.to(device=grad.device, dtype=torch.bool), 0.0)
                            return grad
                        return hook_fn
                    param.register_hook(make_output_hook(freeze_mask))
                    frozen_count = int(freeze_mask.sum().item())
                    frozen_params += frozen_count
                    print(f"  Froze {frozen_count} parameters in output layer '{name}' for {len(token_ids)} special tokens (rows)")
            
            # Handle column-wise layout [hidden, vocab_size]
            elif cols > max_id:
                freeze_mask = torch.zeros_like(param, dtype=torch.bool)
                for token_id in token_ids:
                    tid = int(token_id)
                    if 0 <= tid < cols:
                        freeze_mask[:, tid] = True

                if freeze_mask.any():
                    def make_output_hook(mask):
                        def hook_fn(grad):
                            if grad is not None:
                                return grad.masked_fill(mask.to(device=grad.device, dtype=torch.bool), 0.0)
                            return grad
                        return hook_fn
                    param.register_hook(make_output_hook(freeze_mask))
                    frozen_count = int(freeze_mask.sum().item())
                    frozen_params += frozen_count
                    print(f"  Froze {frozen_count} parameters in output layer '{name}' for {len(token_ids)} special tokens (cols)")
            else:
                print(f"  Warning: Could not map token IDs to output layer '{name}' shape {tuple(param.shape)}")
    
    return frozen_params


def _find_embedding_layers(model):
    """
    Find embedding layers in the model.
    
    Args:
        model: The model to search for embedding layers
    
    Returns:
        list: List of (name, module) tuples for embedding layers
    """
    embedding_layers = []
    
    # Common embedding layer names and types
    embedding_names = ['embed_tokens', 'wte', 'word_embeddings', 'embeddings', 'token_embedding']
    
    for name, module in model.named_modules():
        # Check by name
        module_name_lower = name.lower()
        if any(emb_name in module_name_lower for emb_name in embedding_names):
            if hasattr(module, 'weight'):
                embedding_layers.append((name, module))
                continue
        
        # Check by type
        if isinstance(module, torch.nn.Embedding):
            embedding_layers.append((name, module))
    
    return embedding_layers


def _find_output_projection_layers(model):
    """
    Find output projection layers in the model (typically lm_head).
    
    Args:
        model: The model to search for output projection layers
    
    Returns:
        list: List of (name, module) tuples for output projection layers
    """
    output_layers = []
    
    # Common output layer names
    output_names = ['lm_head', 'output_projection', 'classifier', 'head', 'output_layer']
    
    for name, module in model.named_modules():
        # Check by name
        module_name_lower = name.lower()
        if any(out_name in module_name_lower for out_name in output_names):
            if hasattr(module, 'weight'):
                output_layers.append((name, module))
                continue
            
    return output_layers


def _get_parameter_by_name(model, param_name: str):
    """Get parameter object by its full name."""
    try:
        # Navigate through the model using the parameter name
        obj = model
        for attr in param_name.split('.'):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        print(f"Warning: Could not find parameter {param_name}")
        return None


# =========================
# Lottery Ticket Adaptation (LoTA) Baseline Implementation
# Reference: "Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs"
# Panda et al., 2024 (arXiv:2406.16797)
# =========================

import math
from dataclasses import dataclass

@dataclass
class LotaState:
    """Container for LoTA state.

    Attributes:
        base_weights: Dict[str, torch.Tensor] copy of original pretrained weights (w_P)
        mask: Dict[str, torch.BoolTensor] element-wise trainable mask (True = trainable, False = frozen)
        sparsity: float fraction of weights frozen (e.g. 0.9 => 90% frozen, 10% trainable)
        calibration_steps: int number of calibration update steps used to build mask
        total_params: int total parameter elements considered
        trainable_params: int elements marked trainable by mask
        skip_embeddings_and_head: bool whether embeddings / lm_head skipped in mask selection (fully trainable)
    """
    base_weights: Dict[str, torch.Tensor]
    mask: Dict[str, torch.BoolTensor]
    sparsity: float
    calibration_steps: int
    total_params: int
    trainable_params: int
    skip_embeddings_and_head: bool = False


def lota_calibrate_mask(
    model,
    dataloader,
    optimizer,
    sparsity: float = 0.9,
    calibration_steps: int = 100,
    device: Optional[Union[str, torch.device]] = None,
    skip_embeddings_and_head: bool = False,
    grad_accum_steps: int = 1,
    max_batches: Optional[int] = None,
    verbose: bool = True,
    use_amp: bool = True,
    amp_dtype: str = "bfloat16",
    microbatch_chunks: Optional[int] = None,
    enable_gradient_checkpointing: bool = True,
    disable_cache_during_calibration: bool = True,
) -> LotaState:
    """Phase 1 + Phase 2 of LoTA: Mask calibration & extraction.

    Workflow:
      1. Copy base weights w_P.
      2. Perform `calibration_steps` standard fine-tuning updates on calibration data (can be <= one epoch).
      3. Compute task vector Δ = w_F - w_P.
      4. Build element-wise magnitude mask selecting top (1 - sparsity) fraction of |Δ| as trainable.

    After this, you should call `lota_prepare_sparse_training` to reset weights back to w_P
    and register gradient masking hooks for sparse adaptation.

    Args:
        model: HF model or wrapped model (supports `.named_parameters()`).
        dataloader: iterable yielding batches (expects batch dict with input_ids/labels typical HF).
        optimizer: torch optimizer.
        sparsity: Fraction of weights to freeze (0.9 => keep 10% trainable).
        calibration_steps: Number of optimization steps for calibration (T in paper's mask calibration phase).
        device: Device to move batches to (defaults to first param device).
        skip_embeddings_and_head: Skip embedding & lm_head params from mask selection; they remain fully trainable.
        grad_accum_steps: Gradient accumulation steps.
        max_batches: Optional cap on batches to iterate (overrides dataloader length if set).
        verbose: Print progress information.

    Returns:
        LotaState containing base weights and trainable mask.
    """
    import torch.nn.functional as F
    from contextlib import nullcontext

    actual_model = model
    if device is None:
        device = next(actual_model.parameters()).device

    # Ensure training mode for calibration
    try:
        actual_model.train()
    except Exception:
        pass

    # Optionally turn off KV cache and enable gradient checkpointing during calibration
    original_use_cache = None
    if disable_cache_during_calibration and hasattr(actual_model, 'config') and hasattr(actual_model.config, 'use_cache'):
        original_use_cache = actual_model.config.use_cache
        try:
            actual_model.config.use_cache = False
            if verbose:
                print("[LoTA] Temporarily disabling use_cache during calibration.")
        except Exception:
            pass

    if enable_gradient_checkpointing and hasattr(actual_model, 'gradient_checkpointing_enable'):
        try:
            actual_model.gradient_checkpointing_enable()
            if verbose:
                print("[LoTA] Gradient checkpointing enabled during calibration.")
        except Exception:
            if verbose:
                print("[LoTA] Could not enable gradient checkpointing; continuing without it.")

    # Copy base weights (w_P)
    base_weights: Dict[str, torch.Tensor] = {}
    for name, p in actual_model.named_parameters():
        # store on CPU to reduce device memory pressure
        base_weights[name] = p.detach().cpu().clone()

    if verbose:
        print(f"[LoTA] Starting calibration: steps={calibration_steps}, sparsity={sparsity:.2f}")

    step = 0
    batch_iter = iter(dataloader)
    from collections.abc import Mapping

    fallback_loss_used = False
    while step < calibration_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            # restart if need more steps (allows multiple epochs implicitly)
            batch_iter = iter(dataloader)
            batch = next(batch_iter)
        if max_batches is not None and step >= max_batches:
            break

        # Normalize batch format and move to device (supports Mapping/BatchEncoding or tuple/list)
        if isinstance(batch, Mapping):
            # Convert to a plain dict to allow safe modification
            batch = dict(batch)
            # If labels are missing but input_ids present (common for CLM), use input_ids as labels
            if 'labels' not in batch and 'input_ids' in batch:
                batch['labels'] = batch['input_ids']
            # Ensure we don't request heavy outputs during calibration
            batch.setdefault('use_cache', False)
            batch.setdefault('output_attentions', False)
            batch.setdefault('output_hidden_states', False)
            for k, v in list(batch.items()):
                if hasattr(v, 'to'):
                    batch[k] = v.to(device)
        elif isinstance(batch, (list, tuple)):
            batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]

        # Helper: forward with autocast
        def _autocast_context():
            if not use_amp:
                return nullcontext()
            # On AMD/ROCm torch.cuda.is_available() is True; device_type is 'cuda'
            dtype = torch.bfloat16 if amp_dtype.lower() in {"bf16", "bfloat16"} else torch.float16
            try:
                return torch.autocast(device_type='cuda' if torch.cuda.is_available() else (device.type if hasattr(device, 'type') else 'cpu'), dtype=dtype)
            except Exception:
                return nullcontext()

        def _model_forward(b):
            if not hasattr(actual_model, 'forward'):
                raise RuntimeError("Model does not have forward method for LoTA calibration.")
            if isinstance(b, Mapping):
                return actual_model(**b)
            else:
                return actual_model(b[0], labels=b[1] if len(b) > 1 else None)

        # Micro-batch split to reduce peak memory
        loss = None
        num_chunks = 1
        if microbatch_chunks and isinstance(batch, Mapping) and 'input_ids' in batch and torch.is_tensor(batch['input_ids']) and batch['input_ids'].dim() >= 1:
            bsz = batch['input_ids'].size(0)
            if bsz > 0:
                num_chunks = min(max(1, int(microbatch_chunks)), int(bsz))
        
        if num_chunks > 1 and isinstance(batch, Mapping):
            # Split tensor-like keys along batch dim when first dim matches
            splits = {}
            for k, v in batch.items():
                if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == batch['input_ids'].size(0):
                    splits[k] = v.split((v.size(0) + num_chunks - 1)//num_chunks, dim=0)
                else:
                    # broadcast scalar/others to chunks
                    splits[k] = [v] * num_chunks

            for i in range(num_chunks):
                sub_batch = {k: (splits[k][i] if isinstance(splits[k], tuple) or isinstance(splits[k], list) else splits[k]) for k in splits}
                with _autocast_context():
                    out_i = _model_forward(sub_batch)
                sub_loss = getattr(out_i, 'loss', None)
                if sub_loss is None:
                    # manual fallback loss
                    logits = getattr(out_i, 'logits', None)
                    labels = sub_batch.get('labels', sub_batch.get('input_ids', None))
                    if logits is None or labels is None:
                        raise RuntimeError("Batch forward produced no .loss and fallback could not find logits or labels.")
                    if logits.dim() == 3:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        if shift_labels.dtype != torch.long:
                            shift_labels = shift_labels.to(torch.long)
                        sub_loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100
                        )
                    else:
                        if labels.dim() > 1:
                            labels = labels.view(-1)
                        if labels.dtype != torch.long:
                            labels = labels.to(torch.long)
                        sub_loss = F.cross_entropy(logits, labels)
                    if not fallback_loss_used and verbose:
                        print("[LoTA] Fallback manual loss used (no .loss on outputs).")
                        fallback_loss_used = True
                # Average over chunks and account for grad accumulation
                (sub_loss / (grad_accum_steps * num_chunks)).backward()
            loss = sub_loss  # for logging only (last chunk)
        else:
            with _autocast_context():
                outputs = _model_forward(batch)

            loss = getattr(outputs, 'loss', None)
            if loss is None:
                # Attempt manual causal LM loss computation as fallback
                logits = getattr(outputs, 'logits', None) if not isinstance(outputs, dict) else outputs.get('logits', None)
                labels = None
                if isinstance(batch, dict):
                    labels = batch.get('labels', batch.get('input_ids', None))
                elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                    labels = batch[1]
                elif isinstance(batch, (list, tuple)) and len(batch) == 1:
                    labels = batch[0]

                if logits is None or labels is None:
                    raise RuntimeError("Batch forward produced no .loss and fallback could not find logits or labels for manual loss computation.")

                if logits.dim() == 3:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    if shift_labels.dtype != torch.long:
                        shift_labels = shift_labels.to(torch.long)
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    if labels.dim() > 1:
                        labels = labels.view(-1)
                    if labels.dtype != torch.long:
                        labels = labels.to(torch.long)
                    loss = F.cross_entropy(logits, labels)
                if not fallback_loss_used and verbose:
                    print("[LoTA] Fallback manual loss used (model output lacked .loss).")
                    fallback_loss_used = True

            (loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        if verbose and step % max(1, max(1, calibration_steps) // 10) == 0:
            try:
                ls = float(loss.item())
            except Exception:
                ls = float('nan')
            print(f"[LoTA] Calibration progress {step}/{calibration_steps} (loss={ls:.4f})")

    if verbose:
        print("[LoTA] Calibration finished. Extracting task vector and mask ...")

    # Compute deltas
    deltas: Dict[str, torch.Tensor] = {}
    for name, p in actual_model.named_parameters():
        # Compute on CPU to avoid device mismatch and reduce GPU memory pressure
        deltas[name] = p.detach().to('cpu') - base_weights[name]

    mask, total_params, trainable_params = _lota_extract_mask_from_delta(
        deltas,
        sparsity=sparsity,
        skip_embeddings_and_head=skip_embeddings_and_head,
        skip_fn=lambda param_name: _should_skip_module(param_name, skip_embeddings_and_head),
    )

    if verbose:
        coverage = trainable_params / total_params if total_params else 0
        print(f"[LoTA] Mask built: trainable elements={trainable_params:,}/{total_params:,} ({coverage:.2%})")

    state = LotaState(
        base_weights=base_weights,
        mask=mask,
        sparsity=sparsity,
        calibration_steps=calibration_steps,
        total_params=total_params,
        trainable_params=trainable_params,
        skip_embeddings_and_head=skip_embeddings_and_head,
    )

    # Attach state to model for convenience
    actual_model._lota_state = state

    # Restore model flags toggled for calibration
    if original_use_cache is not None and hasattr(actual_model, 'config') and hasattr(actual_model.config, 'use_cache'):
        try:
            actual_model.config.use_cache = original_use_cache
            if verbose:
                print("[LoTA] Restored use_cache setting after calibration.")
        except Exception:
            pass
    if enable_gradient_checkpointing and hasattr(actual_model, 'gradient_checkpointing_disable'):
        try:
            actual_model.gradient_checkpointing_disable()
        except Exception:
            pass
    return state


def _lota_extract_mask_from_delta(
    deltas: Dict[str, torch.Tensor],
    sparsity: float,
    skip_embeddings_and_head: bool,
    skip_fn,
) -> tuple:
    """Phase 2 of LoTA: build trainable mask from task vector magnitudes.

    We select the top (1 - sparsity) fraction of |Δ| magnitudes globally across all
    parameters (excluding skipped modules) as trainable (mask=True).

    For parameters considered "skipped" (embeddings/head) we mark mask=True everywhere
    so they remain fully trainable per the project's skip semantics.

    Args:
        deltas: Dict mapping param name -> delta tensor (Δ).
        sparsity: Fraction to freeze globally.
        skip_embeddings_and_head: Whether to skip embedding/lm_head params from consideration.
        skip_fn: Callable deciding if param should be skipped entirely.

    Returns:
        (mask_dict, total_params, trainable_params)
    """
    assert 0.0 < sparsity < 1.0, "sparsity should be between 0 and 1 (exclusive)."

    magnitudes = []
    param_refs = []
    total_params = 0

    # First pass: collect magnitudes for non-skipped parameters
    for name, delta in deltas.items():
        if delta is None or delta.numel() == 0:
            continue
        if skip_fn(name):
            continue  # skip from global thresholding; will mark fully trainable later
        flat = delta.view(-1).abs()
        magnitudes.append(flat)
        param_refs.append((name, delta.shape))
        total_params += flat.numel()

    # Edge: if everything is skipped, build all-True masks
    if total_params == 0:
        mask_all = {name: torch.ones(delta.shape, dtype=torch.bool) for name, delta in deltas.items() if delta is not None}
        trainable_params = sum(int(delta.numel()) for delta in deltas.values() if delta is not None)
        return mask_all, trainable_params, trainable_params

    # Concatenate magnitudes for global threshold
    all_mags = torch.cat(magnitudes, dim=0)
    trainable_fraction = 1.0 - sparsity
    k = int(math.floor(trainable_fraction * all_mags.numel()))

    # Build mask dict
    mask: Dict[str, torch.BoolTensor] = {}
    trainable_params = 0

    if k <= 0:
        # Extremely high sparsity: everything non-skipped frozen, skipped fully trainable
        for name, delta in deltas.items():
            if delta is None:
                continue
            if skip_fn(name):
                mask[name] = torch.ones(delta.shape, dtype=torch.bool)
                trainable_params += int(delta.numel())
            else:
                mask[name] = torch.zeros(delta.shape, dtype=torch.bool)
        total = sum(int(delta.numel()) for delta in deltas.values() if delta is not None)
        return mask, total, trainable_params

    # Determine threshold using topk (efficient for large tensors)
    if k >= all_mags.numel():
        threshold = all_mags.min() - 1e-12
    else:
        topk_vals, _ = torch.topk(all_mags, k, largest=True, sorted=False)
        threshold = topk_vals.min()

    # Second pass: construct masks
    offset = 0
    for (name, shape) in param_refs:
        numel = math.prod(shape)
        slice_mags = all_mags[offset: offset + numel]
        offset += numel
        local_mask_flat = slice_mags >= threshold
        mask[name] = local_mask_flat.view(shape)
        trainable_params += int(local_mask_flat.sum().item())

    # Fill skipped and any remaining params
    total = 0
    for name, delta in deltas.items():
        if delta is None:
            continue
        total += int(delta.numel())
        if name in mask:
            continue
        if skip_fn(name):
            mask[name] = torch.ones(delta.shape, dtype=torch.bool)
            trainable_params += int(delta.numel())
        else:
            mask[name] = torch.zeros(delta.shape, dtype=torch.bool)

    return mask, total, trainable_params


def lota_prepare_sparse_training(model, lota_state: Optional[LotaState] = None, verbose: bool = True):
    """Phase 3 of LoTA: reset to base weights and install gradient hooks for sparse training.

    After calibration & mask extraction, we:
      * Reset model weights back to w_P (base).
      * Register gradient hooks per parameter zeroing gradients where mask=False.

    Args:
        model: HF model or wrapper.
        lota_state: Output of `lota_calibrate_mask` (if None, uses model._lota_state).
        verbose: Print statistics.
    """
    actual_model = model
    if lota_state is None:
        lota_state = getattr(actual_model, '_lota_state', None)
    if lota_state is None:
        raise ValueError("LoTA state not found. Run lota_calibrate_mask first.")

    # Reset weights to base (w_P)
    for name, p in actual_model.named_parameters():
        if name in lota_state.base_weights:
            p.data.copy_(lota_state.base_weights[name].to(p.device))

    # Install hooks for sparse training
    for name, p in actual_model.named_parameters():
        mask_tensor = lota_state.mask.get(name)
        if mask_tensor is None:
            continue
        mask_tensor = mask_tensor.to(p.device)
        if mask_tensor.all():
            # fully trainable
            p.requires_grad = True
            continue
        if (~mask_tensor).all():
            # fully frozen
            p.requires_grad = False
            continue

        # Partial: keep requires_grad, but zero-out gradients where mask=False
        def _grad_hook_factory(local_mask: torch.Tensor):
            def _hook(grad: torch.Tensor):
                return grad.masked_fill((~local_mask.to(device=grad.device, dtype=torch.bool)), 0.0)
            return _hook
        p.register_hook(_grad_hook_factory(mask_tensor))

    if verbose:
        tp = lota_state.trainable_params
        tot = lota_state.total_params
        ratio = (tp / tot) if tot else 0.0
        print(f"[LoTA] Sparse training prepared. Trainable elements: {tp:,}/{tot:,} ({ratio:.2%})")

    actual_model._lota_active = True
    return lota_state


def lota_parameter_summary(model):
    """Return summary of LoTA state if active."""
    actual_model = model.model if hasattr(model, 'model') else model
    state = getattr(actual_model, '_lota_state', None)
    if state is None:
        return "LoTA state not initialized."
    return (
        f"LoTA(sparsity={state.sparsity:.2f}, calibration_steps={state.calibration_steps}, "
        f"trainable={state.trainable_params:,}/{state.total_params:,} ({state.trainable_params/state.total_params:.2%})" )

def build_lassu_english_anchor_column_scales(
    english_score_path,
    current_score_path,
    old_score_paths=None,
    en_current_shared_scale=0.7,
    en_past_shared_scale=0.1,
    en_only_scale=0.0,
    others_scale=1.0,
):
    """
    English-Anchored LA-SSU.

    只特殊处理 English-related columns：

        C = EN ∩ CUR
            英文和当前语言共享重要。
            scale = en_current_shared_scale，默认 0.7。

        P = EN ∩ PAST - CUR
            英文和历史语言共享，当前语言不重要。
            scale = en_past_shared_scale，默认 0.1。

        E = EN - CUR - PAST
            只有英文重要。
            scale = en_only_scale，默认 0.0。

        Others
            不属于上面三类。
            scale = others_scale，默认 1.0。

    注意：
        这里不保护 Past-only，也不处理 Current-Past-only。
        这正是简化版思路：只修改英文相关重要 columns。
    """
    if isinstance(old_score_paths, str):
        old_score_paths = [p for p in old_score_paths.split(",") if p.strip()]
    old_score_paths = old_score_paths or []

    print("=" * 80)
    print("[EA-LA-SSU] Building English-anchored column scales")
    print(f"[EA-LA-SSU] english_score_path = {english_score_path}")
    print(f"[EA-LA-SSU] current_score_path = {current_score_path}")
    print(f"[EA-LA-SSU] old_score_paths = {old_score_paths}")
    print(
        "[EA-LA-SSU] scales: "
        f"EN∩CUR={en_current_shared_scale}, "
        f"EN∩PAST-CUR={en_past_shared_scale}, "
        f"EN-only={en_only_scale}, "
        f"others={others_scale}"
    )
    print("=" * 80)

    english_scores = _lassu_score_file_load(english_score_path)
    current_scores = _lassu_score_file_load(current_score_path)
    old_scores_list = [_lassu_score_file_load(p) for p in old_score_paths]

    scale_dict = {}

    for full_param_name, cur_info in current_scores.items():
        if full_param_name == "__meta__":
            continue

        if not isinstance(cur_info, dict):
            continue

        if "top_idx" not in cur_info or "num_units" not in cur_info:
            continue

        if full_param_name not in english_scores:
            print(f"[EA-LA-SSU] Warning: {full_param_name} not found in English scores, skip.")
            continue

        en_info = english_scores[full_param_name]
        if not isinstance(en_info, dict) or "top_idx" not in en_info:
            print(f"[EA-LA-SSU] Warning: invalid English score entry for {full_param_name}, skip.")
            continue

        num_units = int(cur_info["num_units"])
        unit_axis = cur_info.get("unit_axis", "column")

        cur_top = set(cur_info["top_idx"].view(-1).long().tolist())
        en_top = set(en_info["top_idx"].view(-1).long().tolist())

        # 历史语言重要 column 的并集
        past_union = set()
        for old_scores in old_scores_list:
            if full_param_name not in old_scores:
                continue
            old_info = old_scores[full_param_name]
            if not isinstance(old_info, dict) or "top_idx" not in old_info:
                continue

            old_idx = old_info["top_idx"].view(-1).long()
            old_idx = old_idx[(old_idx >= 0) & (old_idx < num_units)]
            past_union.update(old_idx.tolist())

        # 严格三类：
        # C = EN ∩ CUR
        c_set = en_top & cur_top

        # P = EN ∩ PAST - CUR
        p_set = (en_top & past_union) - cur_top

        # E = EN - CUR - PAST
        e_set = en_top - cur_top - past_union

        # 默认 Others 正常更新
        scale = torch.full((num_units,), float(others_scale), dtype=torch.float32)

        # 先设置 E，再设置 P，再设置 C。
        # C 优先级最高，因为当前语言也需要它。
        for j in e_set:
            if 0 <= j < num_units:
                scale[j] = float(en_only_scale)

        for j in p_set:
            if 0 <= j < num_units:
                scale[j] = float(en_past_shared_scale)

        for j in c_set:
            if 0 <= j < num_units:
                scale[j] = float(en_current_shared_scale)

        scale_dict[full_param_name] = {
            "scale": scale,
            "unit_axis": unit_axis,
            "num_units": num_units,
            "param_shape": cur_info.get("param_shape", None),
        }

        print(
            f"[EA-LA-SSU] {full_param_name}: "
            f"C=EN∩CUR={len(c_set)}, "
            f"P=EN∩PAST-CUR={len(p_set)}, "
            f"E=EN-only={len(e_set)}, "
            f"past_union={len(past_union)}, "
            f"cur_top={len(cur_top)}, "
            f"en_top={len(en_top)}"
        )

    print(f"[EA-LA-SSU] Built scale tensors for {len(scale_dict)} parameters")
    return scale_dict


def apply_lassu_english_anchor_from_score_files(
    model,
    english_score_path,
    current_score_path,
    old_score_paths=None,
    en_current_shared_scale=0.7,
    en_past_shared_scale=0.1,
    en_only_scale=0.0,
    others_scale=1.0,
):
    """
    English-Anchored LA-SSU 训练阶段总入口。

    训练时读取：
        English score
        当前语言 score
        历史语言 score list

    然后生成：
        C = EN ∩ CUR
        P = EN ∩ PAST - CUR
        E = EN - CUR - PAST

    最后注册 gradient scale hook。
    """
    scale_dict = build_lassu_english_anchor_column_scales(
        english_score_path=english_score_path,
        current_score_path=current_score_path,
        old_score_paths=old_score_paths,
        en_current_shared_scale=en_current_shared_scale,
        en_past_shared_scale=en_past_shared_scale,
        en_only_scale=en_only_scale,
        others_scale=others_scale,
    )

    return apply_lassu_column_scales(model, scale_dict)