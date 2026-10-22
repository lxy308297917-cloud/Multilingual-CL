"""Retain original vocabulary loss; reduce vocabulary before selecting positions.

Only masked batchmean changes reduction order. No vocabulary component, clipping,
causal position, or valid-token denominator is removed. Other modes fall back.
"""
import functools
import inspect


def make_mask_reduction_loss(original, counts):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        labels = params['labels']
        if labels is not None and params['reduction'] == 'batchmean':
            valid = labels.ne(-100)
            params['labels'] = None
            params['reduction'] = 'none'
            components = original(**params)
            per_position = components.sum(dim=-1)
            counts['position_reduction_calls'] = counts.get('position_reduction_calls', 0) + 1
            return per_position[valid].sum() / valid.sum()
        counts['original_fallback_calls'] = counts.get('original_fallback_calls', 0) + 1
        return original(*args, **kwargs)

    return wrapped
