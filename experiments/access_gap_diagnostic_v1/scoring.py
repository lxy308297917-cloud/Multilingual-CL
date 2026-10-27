"""Frozen explicit-final scoring; no fallback to last letter or last number."""
import re
FINAL=re.compile(r'^\s*Final\s*:\s*([ABCD])\s*[.]?\s*$',re.I|re.M)
DECLARED=re.compile(r'(?:Final(?:\s+answer)?|Answer|Jibu(?:\s+la\s+mwisho)?)\s*:\s*([ABCD])\b',re.I)
PHRASE=re.compile(r'(?:correct\s+answer\s+is|jibu\s+sahihi\s+ni)\s*\(?([ABCD])\)?\b',re.I)
def score(text,gold):
 finals=FINAL.findall(text)
 labels={x.upper() for x in DECLARED.findall(text)+PHRASE.findall(text)}
 answer=finals[0].upper() if len(finals)==1 and len(labels)<=1 else None
 explanation=FINAL.sub('',text).strip()
 return dict(answer=answer,correct=answer==gold,extraction_failure=answer is None,
             conflicting_declarations=len(labels)>1,final_line_count=len(finals),
             format_compliant=answer is not None and bool(explanation),explanation=explanation)
def accept():
 cases=[('A','A',False),('Final: A','A',True),('Explanation.\nFinal: B','B',True),
 ('Answer: C\nFinal: A','A',False),('Final: A\nFinal: B','A',False),
 ('The correct answer is B.\nFinal: A','A',False),('A B C D','A',False),
 ('Final: A\nFinal: A','A',False),('Final: B','A',False),('Maelezo.\nFinal: D','D',True)]
 for text,gold,expected in cases:assert score(text,gold)['correct']==expected,(text,gold)
 return {'passed':True,'controlled_cases':len(cases),'scope':'Explicit final extraction and common contradictory declarations; no semantic verification of all prose.'}
if __name__=='__main__':print(accept())
