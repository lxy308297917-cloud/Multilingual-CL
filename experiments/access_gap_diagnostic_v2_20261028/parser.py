"""Explicit-choice parser. Unparsed is a separate state, never a content diagnosis."""
import re
DECL=re.compile(r'^\s*(?:\*\*)?(?:Final(?:\s+answer)?|Answer|Jibu(?:\s+la\s+mwisho)?)\s*(?:\*\*)?\s*:\s*(?:\*\*)?([ABCD])\b',re.I|re.M)
CLAIM=re.compile(r'\b(?:correct\s+answer\s+is|jibu\s+sahihi\s+ni)\s*\(?([ABCD])\)?\b',re.I)
INITIAL=re.compile(r'^\s*([ABCD])\s*(?:[.)]|$)')
OPTIONS=re.compile(r'(?:^|\s)([ABCD])[.)]\s')
AMBIG=re.compile(r'^\s*(?:[,/&]\s*[ABCD]\b|(?:or|and|au|na)\s+[ABCD]\b)',re.I)
STRICT=re.compile(r'^\s*Final\s*:\s*([ABCD])\s*[.]?\s*$',re.I|re.M)
def parse(text):
 matches=list(DECL.finditer(text));initial=INITIAL.match(text);labels={m.group(1).upper() for m in matches}|{x.upper() for x in CLAIM.findall(text)}
 if initial:labels.add(initial.group(1))
 ambiguous=any(AMBIG.match(text[m.end():]) for m in matches)
 if initial and AMBIG.match(text[initial.end():]):ambiguous=True
 conflicting=len(labels)>1
 answer=None
 if not ambiguous and not conflicting:
  if matches:answer=matches[0].group(1).upper()
  elif initial and len(set(OPTIONS.findall(text)))<=1:answer=initial.group(1)
 strict=STRICT.findall(text)
 lines=text.strip().splitlines()
 strict_final_line=bool(len(strict)==1 and lines and STRICT.fullmatch(lines[-1]) and answer==strict[0].upper())
 explanation=DECL.sub('',text)
 if matches:explanation=text[:matches[0].start()]+text[matches[0].end():]
 elif initial:explanation=text[initial.end():]
 return dict(answer=answer,ambiguous=ambiguous,conflicting=conflicting,unparsed=answer is None,
  strict_independent_Final_line=len(strict)==1 and answer is not None,
  strict_Final_last_line=strict_final_line,explanation=explanation.strip(' .,:*\n'))
CASES=[('Final: B','B'),('Final: B.','B'),('Final: B, because ...','B'),('Final: B. The passage ...','B'),
 ('Answer: B','B'),('Jibu: B','B'),('B. explanation ...','B'),('B','B'),('Jibu la mwisho: C','C'),
 ('Final: A\nAnswer: B',None),('Final: A or B',None),('Final: A, B',None),('A. one B. two C. three D. four',None),
 ('A good sentence.',None),('The passage mentions B but gives no answer.',None),('The example string "Final: B" is only a quote.',None),
 ('No clear answer.',None),('Final: A\nFinal: A','A'),('Explanation.\nFinal: D','D'),('B. text\nFinal: C',None),
 ('B. text says the correct answer is C.',None),('Answer: A/B',None),('Answer: A and C',None)]
def acceptance():
 for text,expected in CASES:assert parse(text)['answer']==expected,(text,parse(text),expected)
 return dict(passed=True,cases=len(CASES),retrospective_only=True)
