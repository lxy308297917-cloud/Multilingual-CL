"""Conservative, latched correctness cache. No confidence-as-correctness shortcut."""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import re


def numeric_value(text):
    text = str(text).strip().replace('−','-')
    frac = re.fullmatch(r'\\(?:d?frac)\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}', text)
    if frac:
        try:return Fraction(int(frac[1]), int(frac[2]))
        except ZeroDivisionError:return None
    if ',' in text:
        if not re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',text):return None
        text = text.replace(',','')
    if re.fullmatch(r'[+-]?\d+/[+-]?\d+',text):
        try:return Fraction(text)
        except (ValueError,ZeroDivisionError):return None
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',text):return None
    try:
        value=Decimal(text)
        return Fraction(value) if value.is_finite() else None
    except (InvalidOperation,ValueError):return None


def extract_answer(text):
    boxes=[]
    for match in re.finditer(r'\\boxed\s*\{',text):
        start=match.end();end=start;depth=1
        while end<len(text) and depth:
            if text[end]=='{':depth+=1
            elif text[end]=='}':depth-=1
            end+=1
        if depth==0:boxes.append(text[start:end-1])
    if boxes:return numeric_value(boxes[-1]),'boxed'
    explicit=re.findall(r'(?:####|Final answer\s*:|Answer\s*:|Jibu\s*:)[ \t]*([^\n]+)',text,re.I)
    if explicit:return numeric_value(explicit[-1].strip().rstrip('.')),'explicit'
    return numeric_value(text),'bare'


def judge(outputs, gold):
    """Outputs are ordered greedy, sample17/18/19. Truncation is a failure."""
    if len(outputs)!=4 or [x['kind'] for x in outputs]!=['greedy','sample','sample','sample'] or [x['seed'] for x in outputs[1:]]!=[17,18,19]:
        raise ValueError('Expected frozen 1+3 generation budget')
    value=numeric_value(gold)
    if value is None:raise ValueError('Unverified or unsupported numeric gold')
    correct=[]
    for row in outputs:
        pred,_=extract_answer(row['text'])
        correct.append(not row['capped'] and pred is not None and pred==value)
    return bool(correct[0] and sum(correct[1:])>=2),correct


class MasteryCache:
    def __init__(self, data_sha256, prompt_sha256):
        self.identity={'data_sha256':data_sha256,'prompt_sha256':prompt_sha256}
        self.records={};self.round=-1

    def refresh(self, results, *, round_id, model_identity):
        if round_id!=self.round+1:raise ValueError('Nonsequential or repeated refresh')
        if self.records and set(results)!=set(self.records):raise ValueError('Mastery pool changed')
        updates={}
        for key,result in results.items():
            mastered,correct=judge(result['outputs'],result['gold'])
            previous=self.records.get(key)
            if previous and result['gold']!=previous['gold']:raise ValueError('Gold changed')
            base=mastered if round_id==0 else previous['base']
            streak=int(mastered)+(previous['streak'] if previous and mastered else 0)
            acquired=(not base) and ((previous['acquired'] if previous else False) or (round_id>0 and streak>=2))
            updates[key]={'gold':result['gold'],'base':base,'acquired':acquired,'known':base or acquired,
                          'current_pass':mastered,'streak':streak,'correct':correct,'outputs':result['outputs'],
                          'model_identity':model_identity,'round_id':round_id}
        self.records=updates;self.round=round_id
        return {k:v['known'] for k,v in updates.items()}

    def state_dict(self):return {'identity':self.identity,'round':self.round,'records':self.records}
    def load_state_dict(self,state):
        if state['identity']!=self.identity:raise ValueError('Mastery identity mismatch')
        self.records=state['records'];self.round=state['round']
