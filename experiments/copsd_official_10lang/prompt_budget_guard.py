"""Fail before rollout if its prompt leaves no room for the fixed generation budget.

Does not truncate inputs or alter sampling, packing, teacher or KD loss. Installed
only by the separate repaired-data entry, never in already running experiments.
"""
from vllm_update_batch import UpdateBatch


def install():
    original = UpdateBatch.prefetch

    def prefetch(self, *args, **kwargs):
        batches, count = original(self, *args, **kwargs)
        if not self.prompts:
            return batches, count
        t = self.trainer
        context = self.engine.llm_engine.model_config.max_model_len
        completion = t.args.max_completion_length
        lengths = [len(x) for x in t.processing_class(self.prompts,
                    add_special_tokens=False)['input_ids']]
        for n in lengths:
            if n + completion > context:
                raise ValueError(f'Rollout prompt uses {n} tokens, context={context}, '
                                 f'completion budget={completion}; reject before vLLM generation')
        return batches, count

    UpdateBatch.prefetch = prefetch
