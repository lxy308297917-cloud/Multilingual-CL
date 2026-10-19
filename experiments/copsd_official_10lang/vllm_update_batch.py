"""Batch only the rollouts within one optimizer update; preserve official packing/KD.

Independent adaptation, not enabled in existing v1 runs. A real full-length GPU
probe, resume and ordinary exit are required before formal use. Sampling streams
can change with batching: equal seeds do not imply identical stochastic outputs.
"""
import json
import time
from types import MethodType


class UpdateBatch:
    def __init__(self, trainer):
        self.trainer = trainer
        self.engine = trainer.vllm_engine
        self.raw_generate = self.engine.generate
        self.raw_sleep = self.engine.sleep
        self.raw_prefetch = trainer.get_batch_samples
        self.raw_wake = trainer._wake_vllm_if_needed
        self.prompts = []
        self.outputs = None
        self.position = 0
        self.step = None
        self.params = None
        self.records = []

    def assert_drained(self):
        assert not self.prompts or self.position == len(self.prompts), 'Unconsumed rollout group'

    def prefetch(self, epoch_iterator, num_batches, device):
        self.assert_drained()
        batches, count = self.raw_prefetch(epoch_iterator, num_batches, device)
        t = self.trainer
        assert 0 <= len(batches) <= t.args.gradient_accumulation_steps
        assert not t.reason_first, 'Reason-first is outside this adaptation'
        self.prompts = []
        for batch in batches:
            texts = t.processing_class.batch_decode(batch['student_prompts'], skip_special_tokens=False)
            assert len(texts) == 1, 'Only microbatch1 is supported'
            if t.processing_class.pad_token:
                texts = [s.replace(t.processing_class.pad_token, '') for s in texts]
            self.prompts.extend(texts)
        self.outputs = None
        self.position = 0
        self.params = None
        self.step = t.state.global_step
        return batches, count

    def wake(self):
        # Cached RequestOutputs are on CPU. Keep vLLM weights/KV asleep during KD.
        if self.outputs is None:
            self.raw_wake()

    def generate(self, prompts, sampling_params, use_tqdm=False):
        t = self.trainer
        assert t.state.global_step == self.step, 'Cached rollout crossed an optimizer update'
        assert t._last_vllm_sync_step == self.step, 'Inference weights are stale'
        assert self.position < len(self.prompts), 'No prefetched rollout expected'
        assert prompts == [self.prompts[self.position]], 'Rollout prompt order mismatch'
        assert sampling_params.n == 1 and sampling_params.max_tokens == 2048
        signature = repr(sampling_params)
        if self.outputs is None:
            start = time.monotonic()
            self.outputs = self.raw_generate(self.prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
            assert len(self.outputs) == len(self.prompts)
            assert all(len(o.outputs) == 1 for o in self.outputs)
            self.params = signature
            row = {'global_step': self.step, 'rollouts': len(self.prompts),
                   'generated_tokens': sum(len(o.outputs[0].token_ids) for o in self.outputs),
                   'generation_seconds': time.monotonic() - start}
            self.records.append(row)
            print('UPDATE_BATCH_GENERATION', json.dumps(row), flush=True)
        assert signature == self.params, 'Sampling configuration changed within accumulation'
        result = [self.outputs[self.position]]
        self.position += 1
        return result

    def sleep(self, level=1):
        if not self.engine.llm_engine.model_executor.is_sleeping:
            return self.raw_sleep(level=level)

    def install(self):
        controller = self
        self.trainer.get_batch_samples = MethodType(
            lambda unused, *args, **kwargs: controller.prefetch(*args, **kwargs), self.trainer)
        self.trainer._wake_vllm_if_needed = MethodType(lambda unused: controller.wake(), self.trainer)
        self.engine.generate = self.generate
        self.engine.sleep = self.sleep

    def detach(self):
        # Break bound-method/closure cycles before the validated CUDA pool cleanup.
        self.assert_drained()
        self.trainer.get_batch_samples = self.raw_prefetch
        self.trainer._wake_vllm_if_needed = self.raw_wake
        self.engine.generate = self.raw_generate
        self.engine.sleep = self.raw_sleep
        self.outputs = None
        self.engine = None
        self.trainer = None
        self.raw_generate = self.raw_sleep = self.raw_prefetch = self.raw_wake = None
