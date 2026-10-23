from settings import *
import sys
sys.path.append(str(OLD.parent/'paired_capability_sparse_v1'))
from reading_scoring import sw_score,reading_score,behavior
from aux_scoring import score_record

def scoring(row,text):
 if row['study_partition']=='sw_reading':return sw_score(text,row['reference'])
 if row['study_partition']=='en_reading':return reading_score(text,row['reference'],row['options'])
 return score_record(row['study_partition'],row,text)

def inputs():
 m=json.loads((RUN/'EVALUATION_DATA_FROZEN.json').read_text());assert sha(m['data_path'])==m['data_sha256'];assert sha(REPORT/'code/reading_scoring.py')==m['accepted_reading_functions_sha256'];assert sha(REPORT/'code/aux_scoring.py')==m['aux_scoring_byte_identical_sha256'];return m,rows(m['data_path'])
