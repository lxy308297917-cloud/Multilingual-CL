"""Run the unchanged v2 inference worker with full-budget model identities."""
from pathlib import Path
import importlib.util,sys,runpy,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'outputs/access_gap_fullbudget_followup_20261028'
protocol=json.loads((REPORT/'FOLLOWUP_PROTOCOL_FROZEN.json').read_text())
original=Path(protocol['worker_source']);original_code=original.parent
assert hashlib.sha256(original.read_bytes()).hexdigest()==protocol['worker_sha256']
spec=importlib.util.spec_from_file_location('settings',original_code/'settings.py')
settings=importlib.util.module_from_spec(spec);sys.modules['settings']=settings;spec.loader.exec_module(settings)
settings.RUN=ROOT/'work/access_gap_fullbudget_followup_20261028'
settings.MODELS=tuple(protocol['methods'])
assert settings.sha(settings.RUN/'PROMPTS_PRIVATE.json')==protocol['prompts_sha256']
assert settings.sha(settings.REPORT/'PROTOCOL_FROZEN.json')==protocol['original_protocol_sha256']
sys.path.insert(0,str(original_code))
runpy.run_path(str(original),run_name='__main__')
