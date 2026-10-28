"""Verify a full logical training log, including explicit checkpoint continuations.

Old attempts are retained. A segment discarded after a resume boundary cannot
silently contribute to the endpoint's scientific trajectory.
"""
from pathlib import Path
import json,hashlib

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def verified_segments(default_log, done, method, config_sha):
 if 'continuation_manifest' not in done:
  return [dict(path=Path(default_log),first_step=1,last_step=done['steps'])]
 manifest_path=Path(done['continuation_manifest'])
 assert file_sha(manifest_path)==done['continuation_manifest_sha256']
 manifest=json.loads(manifest_path.read_text())
 assert manifest['method']==method and manifest['config_sha256']==config_sha
 assert manifest['final_step']==done['steps']
 result=[];expected_first=1
 for segment in manifest['segments']:
  path=Path(segment['log_path'])
  assert file_sha(path)==segment['log_sha256']
  first,last=segment['first_step'],segment['last_step']
  assert first==expected_first and last>=first
  if first>1:
   checkpoint=Path(segment['resume_checkpoint']);receipt=checkpoint/'RECOVERABLE.json'
   assert file_sha(receipt)==segment['resume_receipt_sha256']
   saved=json.loads(receipt.read_text());assert saved['step']==first-1
   for rel,digest in saved['files'].items():assert file_sha(checkpoint/rel)==digest
  result.append(dict(path=path,first_step=first,last_step=last))
  expected_first=last+1
 assert expected_first==done['steps']+1
 return result

def logical_records(segments):
 for segment in segments:
  expected=segment['first_step'];last=segment['last_step']
  with segment['path'].open() as stream:
   for line in stream:
    row=json.loads(line)
    if row['step']<segment['first_step']:continue
    if row['step']>last:break
    assert row['step']==expected,(str(segment['path']),expected,row['step'])
    expected+=1
    yield row
  assert expected==last+1,(str(segment['path']),expected,last)
