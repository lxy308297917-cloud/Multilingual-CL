"""Adapt TRL0.26 client schema and batch one unchanged COPSD update's rollouts.

No student/teacher prompt, response budget, or loss is changed. Cached responses
cannot cross an optimizer update. This module has no CUDA/import-time side effects.
"""
import hashlib
import json
import os
import time
from types import MethodType


def compatible_client(base):
    class Client(base):
        def __init__(self, *args, **kwargs):
            kwargs['group_port'] = int(os.environ['COPSD_SERVER_GROUP_PORT'])
            super().__init__(*args, **kwargs)

        def generate(self, prompts, presence_penalty=0., **kwargs):
            assert kwargs.get('n', 1) == 1
            assert kwargs.get('guided_decoding_regex') is None
            additional = dict(kwargs.pop('generation_kwargs', {}) or {})
            assert 'presence_penalty' not in additional
            additional['presence_penalty'] = presence_penalty
            response = super().generate(prompts=prompts, generation_kwargs=additional, **kwargs)
            assert isinstance(response, dict) and 'completion_ids' in response
            completions = response['completion_ids']
            assert len(completions) == len(prompts)
            assert all(isinstance(row, list) and all(isinstance(t, int) for t in row) for row in completions)
            return completions
    return Client


class ServerUpdateBatch:
    def __init__(self, trainer, output):
        self.trainer = trainer
        self.output = output
        self.raw_generate = trainer.vllm_client.generate
        self.raw_prefetch = trainer.get_batch_samples
        self.prompts = []
        self.position = 0
        self.outputs = None
        self.step = None
        self.params = None
        self.records = []

    def assert_drained(self):
        assert not self.prompts or self.position == len(self.prompts), 'Unconsumed rollout group'

    def prefetch(self, *args, **kwargs):
        self.assert_drained()
        batches, count = self.raw_prefetch(*args, **kwargs)
        self.prompts = []
        for batch in batches:
            texts = self.trainer.processing_class.batch_decode(batch['student_prompts'], skip_special_tokens=False)
            assert len(texts) == 1
            pad = self.trainer.processing_class.pad_token
            self.prompts.extend(text.replace(pad, '') if pad else text for text in texts)
        assert len(self.prompts) <= self.trainer.args.gradient_accumulation_steps
        self.position = 0
        self.outputs = None
        self.params = None
        self.step = self.trainer.state.global_step
        return batches, count

    def generate(self, prompts, **kwargs):
        t = self.trainer
        assert t.state.global_step == self.step == t._last_vllm_sync_step, 'Stale rollout policy'
        assert self.position < len(self.prompts)
        assert prompts == [self.prompts[self.position]], 'Rollout prompt order mismatch'
        assert kwargs['n'] == 1 and kwargs['max_tokens'] == 2048
        assert kwargs['temperature'] == 1.1 and kwargs['top_p'] == .95 and kwargs['top_k'] == 20
        signature = json.dumps(kwargs, sort_keys=True)
        if self.outputs is None:
            tick = time.monotonic()
            self.outputs = self.raw_generate(prompts=self.prompts, **kwargs)
            assert len(self.outputs) == len(self.prompts)
            assert all(0 < len(row) <= 2048 for row in self.outputs)
            self.params = signature
            row = {'global_step': self.step, 'rollouts': len(self.prompts),
                   'generated_tokens': sum(map(len, self.outputs)),
                   'generation_seconds': time.monotonic()-tick}
            self.records.append(row)
            with (self.output/'SERVER_ROLLOUTS.PRIVATE.jsonl').open('a') as stream:
                for prompt, ids in zip(self.prompts, self.outputs):
                    stream.write(json.dumps({'global_step': self.step,
                        'student_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'completion_ids': ids})+'\n')
            print('SERVER_BATCH_GENERATION', json.dumps(row), flush=True)
        assert signature == self.params, 'Changed sampling within an update'
        result = [self.outputs[self.position]]
        self.position += 1
        return result

    def install(self):
        controller = self
        self.trainer.get_batch_samples = MethodType(lambda unused,*a,**k: controller.prefetch(*a,**k), self.trainer)
        self.trainer.vllm_client.generate = self.generate

    def detach(self):
        self.assert_drained()
        self.trainer.get_batch_samples = self.raw_prefetch
        self.trainer.vllm_client.generate = self.raw_generate
