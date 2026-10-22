"""Accept one frozen native4B throughput pair; never replace scientific scores."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();code=Path(__file__).parent
    freeze=code/'eval_concurrency4b_frozen_20261022.PUBLIC.json';f=read(freeze)
    for name,digest in f['code_sha256'].items():assert sha(code/name)==digest,name
    state=read(a.root/'gpu-queue/queues/copsd-eval-native4b-bench-v1/state.json');job=state['job']
    assert state['phase']=='completed' and job['exit_code']==0
    assert job['script_sha256']==f['code_sha256']['run_eval_concurrency4b.sh']
    assert len(job['gpu_uuids'])==1 and job['config']['max_user_gpus']==4
    folder=a.root/f['output_relative'];summary=read(folder/'SUMMARY.PUBLIC.json');modes={};rows={}
    assert summary['same_GPU']==job['gpu_uuids'][0] and summary['new_generation_responses']==128
    for mode in [16,32]:
        path=folder/f'concurrency{mode}';cfg=read(path/'CONFIG.json');metrics=read(path/'DONE.json')
        assert cfg['CUDA_VISIBLE_DEVICES']==job['gpu_uuids'][0] and cfg['actual_concurrency_requested']==mode
        assert cfg['reference']['model_revision']==f['model_revision'] and cfg['reference']['model']==str(a.root/'models/Qwen3-4B')
        assert sha(Path(cfg['reference']['adapter'])/'adapter_model.safetensors')==f['adapter_sha256']
        r=[json.loads(line) for line in (path/'ITEMS.jsonl').read_text().splitlines()]
        assert len(r)==len({row['id'] for row in r})==metrics['items']==64
        assert {lang:sum(x['language']==lang for x in r) for lang in ['sw','en']}=={'sw':32,'en':32}
        assert all(x['finish_reason'] in ['stop','length'] and len(x['token_ids'])<=8192 for x in r)
        tokens=sum(len(x['token_ids']) for x in r);assert tokens==metrics['generated_tokens']
        assert sha(path/'ITEMS.jsonl')==metrics['items_sha256']
        assert abs(metrics['tokens_per_generation_second']-tokens/metrics['generation_seconds'])<1e-9
        readings=[]
        for line in (path/'gpu.csv').read_text().splitlines():
            values=line.split(',')
            if len(values)==4:
                try:readings.append([float(x) for x in values[1:]])
                except ValueError:pass
        assert readings and min(x[1] for x in readings)==metrics['minimum_sampled_GPU_free_MiB']
        assert max(x[0] for x in readings)==metrics['sampled_physical_GPU_peak_MiB']
        modes[str(mode)]=metrics;rows[mode]=r
    for x,y in zip(rows[16],rows[32]):assert all(x[k]==y[k] for k in ['id','source_id','language','question','answer','prompt','seed'])
    ratio=modes['16']['generation_seconds']/modes['32']['generation_seconds']
    free=modes['32']['minimum_sampled_GPU_free_MiB']
    hours=(datetime.fromisoformat(job['finished_at'])-datetime.fromisoformat(job['started_at'])).total_seconds()/3600
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tb=a.root/'logs/copsd_official_10lang/tensorboard/native4b_throughput16_32_20261022';writer=SummaryWriter(str(tb))
    for mode in ['16','32']:
        for name in ['tokens_per_generation_second','samples_per_generation_second','generation_seconds','minimum_sampled_GPU_free_MiB','sampled_physical_GPU_peak_MiB']:writer.add_scalar(mode+'/'+name,modes[mode][name],0)
    writer.add_scalar('cost/parent_GPU_hours',hours,0);writer.close();acc=EventAccumulator(str(tb)).Reload();assert len(acc.Tags()['scalars'])==11
    result={'protocol':f['protocol'],'completed_at':datetime.now().isoformat(),'modes':modes,
        'generation_wall_speedup32_vs16':ratio,'measured_same_GPU':job['gpu_uuids'][0],
        'exact_token_sequence_matches':sum(x['token_ids']==y['token_ids'] for x,y in zip(rows[16],rows[32])),
        'candidate_for_future_uniform_protocol':ratio>1 and free>=f['minimum_sampled_free_MiB_for_future_adoption'],
        'threshold_minimum_sampled_free_MiB':f['minimum_sampled_free_MiB_for_future_adoption'],
        'new_generated_tokens':sum(x['generated_tokens'] for x in modes.values()),'parent_GPU_hours':hours,
        'TensorBoard_ten_performance_series_one_cost_read_back':True,
        'scope':'One two-mode engineeringtrial,64 SW32/EN32 full8192 prompts permode; no answergrading or qualityequivalence. Generatedlengths may differ; reporttoken andsamplethroughput. Sharedserver backgroundandmodeorder uncontrolled. Existingseed42/seed43 scientificprotocols remainnative16; futurecomparisons mustuseone uniformlyfixed adoptedprotocol, neverrelabel oldscores.',
        'frozen_sha256':sha(freeze),'summarizer_sha256':sha(Path(__file__))}
    with a.output.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'speedup':ratio,'free_MiB':free,'future_candidate':result['candidate_for_future_uniform_protocol']}))


if __name__=='__main__':main()
