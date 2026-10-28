"""Conservative explicit-final numeric parsing; no gold-dependent or last-number fallback."""
import re
from fractions import Fraction

def numeric(text):
 s=text.strip().replace('−','-').replace(',','')
 s=re.sub(r'\\(?:left|right)','',s)
 for left,right in ((r'\(',r'\)'),(r'\[',r'\]'),('$','$')):
  if s.startswith(left) and s.endswith(right):s=s[len(left):-len(right)].strip()
 s=s.strip().removesuffix('.').strip()
 frac=re.fullmatch(r'\\(?:dfrac|tfrac|frac)\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}',s)
 if frac:
  try:return Fraction(frac[1])/Fraction(frac[2])
  except ZeroDivisionError:return None
 if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?',s):return None
 try:
  if '/' in s:
   a,b=s.split('/');return Fraction(a.strip())/Fraction(b.strip())
  return Fraction(s)
 except (ValueError,ZeroDivisionError):return None

def boxes(text):
 result=[]
 for match in re.finditer(r'\\boxed\s*\{',text):
  depth=1;i=match.end();start=i
  while i<len(text) and depth:
   if text[i]=='{':depth+=1
   elif text[i]=='}':depth-=1
   i+=1
  result.append(text[start:i-1] if depth==0 else None)
 return result

def parse(text):
 candidates=[];unsupported=[]
 for value in re.findall(r'(?m)^\s*####\s*([^\n]+)\s*$',text):
  if re.fullmatch(r'(?i)(?:final\s+)?answer\s*:',value.strip()):continue
  parsed=numeric(value)
  if parsed is None:unsupported.append(dict(signal='hash_final',text=value))
  else:candidates.append((parsed,'hash_final'))
 for value in boxes(text):
  parsed=numeric(value) if value is not None else None
  if parsed is None:unsupported.append(dict(signal='boxed',text=value))
  else:candidates.append((parsed,'boxed'))
 # Bold output is considered only in the final paragraph with a conclusion cue.
 final=text.strip().split('\n\n')[-1].strip()
 if re.search(r'(?i)\b(?:so|thus|therefore|hence|answer|total|is|are)\b',final):
  bold=re.findall(r'\*\*([^*\n]+)\*\*',final)
  for value in bold:
   parsed=numeric(value)
   if parsed is not None:candidates.append((parsed,'terminal_bold'))
 # Explicit final-answer line; whole expression must parse, never search its last digit.
 for value in re.findall(r'(?im)^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:|is|:)\s*(.+?)\s*$',text):
  parsed=numeric(value)
  if parsed is not None:candidates.append((parsed,'explicit_answer_line'))
 values={value for value,_ in candidates}
 if len(values)>1:return None,dict(status='conflicting_explicit_final_answers',signals=[s for _,s in candidates])
 if unsupported:return None,dict(status='unsupported_explicit_final_expression',unsupported=unsupported)
 if len(values)==1:
  value=next(iter(values));return value,dict(status='parsed',signals=[s for _,s in candidates])
 return None,dict(status='no_unambiguous_explicit_numeric_final_answer')

if __name__=='__main__':
 fixtures=[('reason 17\n#### 18','18'),(r'The answer is \boxed{18}','18'),
  (r'\boxed{\frac{1}{2}}','1/2'),('So, the total is **30**.','30'),
  ('#### 18\n#### 19',None),('#### 18\n'+r'\boxed{19}',None),
  ('I used 18 eggs and had 19 left.',None),(r'\boxed{18 km}',None),
  ('The answer is: 18','18'),('So, either **3** or **8**.',None),
  (r'\boxed{\frac{1}{0}}',None),('#### Final Answer:\n'+r'\boxed{12}','12')]
 for text,gold in fixtures:
  value,_=parse(text);assert (str(value) if value is not None else None)==gold,(text,value,gold)
 print('Explicit-final parser acceptance passed',len(fixtures))
