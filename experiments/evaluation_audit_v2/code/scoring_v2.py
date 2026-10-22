"""Conservative content scoring. Format and extraction status remain separate.

No model output is used to tune this module after ACCEPTANCE_FROZEN.json.
Numeric equivalence is exact rational equivalence, not arbitrary expression eval.
Units are recorded; absent unit gold is not evidence of unit correctness.
"""
import re, unicodedata, collections
from fractions import Fraction

def clean(s):
    return unicodedata.normalize('NFKC',s).replace('−','-').replace('–','-').strip()

def tokens(s):
    return ''.join(' ' if unicodedata.category(c).startswith(('P','S')) else c for c in clean(s).casefold()).split()

def sw_score(text, gold):
    # Strip only one conventional answer label, never select a best subspan.
    text=re.sub(r'^\s*(?:jibu|answer)\s*:\s*','',text,flags=re.I)
    a,b=tokens(text),tokens(gold); overlap=sum((collections.Counter(a)&collections.Counter(b)).values())
    return {'content_f1':2*overlap/(len(a)+len(b)) if a or b else 1.,'content_exact':int(a==b),'status':'scored'}

def reading_score(text, gold, options):
    s=clean(text); letters=set()
    # Bare option headings, explicit answer statements and standalone letters.
    for m in re.finditer(r'(?:^|\n)\s*\(?([A-D])\)?(?:[.):]|\s*$)',s): letters.add(m[1])
    for m in re.finditer(r'\b(?:answer|option|choice)\s*(?:is\s*)?[:：]?\s*\(?([A-D])\)?\b',s,re.I): letters.add(m[1].upper())
    for m in re.finditer(r'\b(?:but\s+(?:actually\s+)?|actually\s*[:,]?\s*|instead\s*[:,]?\s*)\(?([A-D])\)?\b',s,re.I):letters.add(m[1].upper())
    if re.fullmatch(r'\(?[A-Da-d]\)?[.!]?',s): letters.add(re.search('[A-Da-d]',s)[0].upper())
    norm=lambda x:' '.join(tokens(x))
    option_matches=[chr(65+i) for i,opt in enumerate(options) if norm(s)==norm(opt)]
    letters.update(option_matches)
    # A correct letter followed by another option's exact text is contradictory.
    rest=re.sub(r'^(?:(?:the\s+)?(?:answer|option|choice)\s*(?:is\s*)?[:：]?\s*)?\(?[A-D]\)?[.):]?\s*','',s,flags=re.I)
    if rest!=s:
        letters.update(chr(65+i) for i,opt in enumerate(options) if norm(rest)==norm(opt))
    # Listing multiple complete options or ambiguous conjunctions must not pass.
    if re.search(r'\b[A-D]\s*(?:or|and|/|,)\s*[A-D]\b',s):
        return {'content_correct':0,'format_correct':int(s==gold),'status':'ambiguous','parsed':None}
    if len(letters)>1: status='ambiguous'; parsed=None
    elif len(letters)==1: status='extracted'; parsed=next(iter(letters))
    else: status='extraction_failed'; parsed=None
    return {'content_correct':int(parsed==gold),'format_correct':int(s==gold),'status':status,'parsed':parsed}

NUM=r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

def numeric(s):
    s=clean(s).replace('\\,','').replace('\\!','').strip()
    # Paired dollars denote TeX math; a single leading dollar is a currency unit.
    if s.startswith('$') and s.endswith('$') and len(s)>1:s=s[1:-1]
    currency=bool(re.match(r'(?i)^(?:usd\s*|\$\s*)',s))
    s=re.sub(r'\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}',r'\1/\2',s)
    s=s.replace('\\%','%'); unit=''
    s=re.sub(r'(?i)^(?:usd\s*|\$\s*)','',s)
    m=re.fullmatch(r'('+NUM+r'(?:\s*/\s*'+NUM+r')?)\s*(%|[A-Za-z]+(?:[ -][A-Za-z]+){0,2})?[.!]?',s)
    if not m:return None,None
    expr=m[1].replace(',','').replace(' ','');unit=m[2] or ''
    if unit.casefold() in {'or','and','because','approximately','maybe','either','wrong','incorrect'}:return None,None
    try:
        if '/' in expr:
            a,b=expr.split('/');value=Fraction(a)/Fraction(b)
        else:value=Fraction(expr)
        return value,unit or ('dollars' if currency else '')
    except (ValueError,ZeroDivisionError):return None,None

def boxed_values(text):
    found=[]
    for m in re.finditer(r'\\boxed\s*\{',text):
        depth=1;start=m.end();i=start
        while i<len(text) and depth:
            depth+=(text[i]=='{')-(text[i]=='}');i+=1
        found.append(text[start:i-1] if depth==0 else '')
    return found

def unit_info(unit):
    u=unit.casefold().strip().replace('u.s.','').strip()
    aliases={'dollar':('currency_usd',Fraction(1)),'dollars':('currency_usd',Fraction(1)),'usd':('currency_usd',Fraction(1)),'cent':('currency_usd',Fraction(1,100)),'cents':('currency_usd',Fraction(1,100)),
             'second':('time',Fraction(1)),'seconds':('time',Fraction(1)),'s':('time',Fraction(1)),'minute':('time',Fraction(60)),'minutes':('time',Fraction(60)),'hour':('time',Fraction(3600)),'hours':('time',Fraction(3600)),
             'meter':('length',Fraction(1)),'meters':('length',Fraction(1)),'m':('length',Fraction(1)),'kilometer':('length',Fraction(1000)),'kilometers':('length',Fraction(1000)),'km':('length',Fraction(1000)),'centimeter':('length',Fraction(1,100)),'centimeters':('length',Fraction(1,100)),'cm':('length',Fraction(1,100)),
             'mile':('length',Fraction('1609.344')),'miles':('length',Fraction('1609.344')),'foot':('length',Fraction('0.3048')),'feet':('length',Fraction('0.3048')),'inch':('length',Fraction('0.0254')),'inches':('length',Fraction('0.0254')),
             'gram':('mass',Fraction(1)),'grams':('mass',Fraction(1)),'g':('mass',Fraction(1)),'kilogram':('mass',Fraction(1000)),'kilograms':('mass',Fraction(1000)),'kg':('mass',Fraction(1000)),'pound':('mass',Fraction('453.59237')),'pounds':('mass',Fraction('453.59237')),
             'month':('calendar',Fraction(1)),'months':('calendar',Fraction(1)),'year':('calendar',Fraction(12)),'years':('calendar',Fraction(12)),
             'square feet':('area',Fraction('0.09290304')),'square foot':('area',Fraction('0.09290304')),'square inches':('area',Fraction('0.00064516')),'square inch':('area',Fraction('0.00064516')),'square yards':('area',Fraction('0.83612736')),'square yard':('area',Fraction('0.83612736')),
             '%':('ratio',Fraction(1,100)),'percent':('ratio',Fraction(1,100)),'ratio':('ratio',Fraction(1)),'fraction':('ratio',Fraction(1))}
    if u in aliases:return aliases[u]
    count_alias={'people':'person','persons':'person','children':'child','boxes':'box','pieces':'piece'}
    u=count_alias.get(u,u[:-1] if u.endswith('s') else u)
    count_words={'count','person','child','piece','doll','stripe','flower','shirt','fish','skip','cat','cake','sock','pair of socks','pairs of socks','animal','wall','door','parent','orange','brick','point','dish','staple','stapler','chicken','pack','member','box','book','apple','banana','bird','bottle','gummy','vitamin','big cat','big cats','student','ticket'}
    if u in count_words:return 'count',Fraction(1)
    return 'unknown:'+u,Fraction(1)

def math_score(text,gold,expected_unit=''):
    s=clean(text); candidates=boxed_values(s); methods=['boxed']*len(candidates)
    for line in s.splitlines():
        m=re.search(r'(?:####|\bfinal\s+answer\s*(?:is|=|:)?|\bthe\s+answer\s+(?:is|=)|^\s*answer\s*:)\s*(.*)',line,re.I)
        if m:
            part=m[1].strip()
            # Same marker surrounding an already extracted box is redundant.
            if '\\boxed' not in part:candidates.append(part);methods.append('explicit_final')
    if candidates and '\n' in s:
        last=s.splitlines()[-1].strip()
        if numeric(last)[0] is not None and last not in candidates:
            candidates.append(last);methods.append('standalone_final_line_alongside_explicit')
    if not candidates:
        # A whole response or isolated final line containing one quantity is OK;
        # an arbitrary last number inside reasoning is never selected.
        last=s.splitlines()[-1].strip() if s else ''
        if numeric(last)[0] is not None:candidates=[last];methods=['standalone_final_line']
    values=[numeric(x) for x in candidates];goldv,goldu=numeric(str(gold))
    if goldv is None:raise ValueError('Unsupported gold numeric expression')
    goldu=expected_unit or goldu
    parsed=set()
    for v,u in values:
        if v is None:continue
        if goldu and u:
            gd,gf=unit_info(goldu);ud,uf=unit_info(u)
            if gd==ud:v=v*uf/gf
        parsed.add(v)
    if not candidates:status='extraction_failed'
    elif any(v is None for v,u in values) or len(parsed)!=1:status='ambiguous'
    else:status='extracted'
    value=next(iter(parsed)) if status=='extracted' else None
    units={u.casefold() for v,u in values if v is not None and u}
    unit_ok=True;numeric_correct=value==goldv;unit_status='implicit_question_unit' if goldu else 'no_unit_gold'
    if status=='extracted' and units:
        if not goldu:
            unit_ok=False;unit_status='unit_needs_review'
        else:
            gd,gf=unit_info(goldu);converted=[]
            for v,u in values:
                ud,uf=unit_info(u or goldu);converted.append((ud,v*uf))
            unit_ok=all(ud==gd for ud,val in converted);numeric_correct=all(val==goldv*gf for ud,val in converted);unit_status='verified_conversion' if unit_ok else 'unit_needs_review' if any(ud.startswith('unknown:') for ud,val in converted) else 'incompatible_unit'
    return {'content_correct':int(numeric_correct and unit_ok),'numeric_value_match_without_unit':int(value==goldv),'format_boxed':int(bool(boxed_values(s))),
            'status':status,'parsed':str(value) if value is not None else None,'method':methods,
            'units':sorted(units),'expected_unit':goldu,'unit_status':unit_status,'unit_gold_available':bool(goldu),'unit_correctness_verified':bool(goldu and units and unit_ok)}

def score(row,text):
    if row['task']=='en_math':return math_score(text,row['reference'],row.get('expected_unit',''))
    if row['task']=='en_reading':return reading_score(text,row['reference'],row['options'])
    if row['task']=='sw_reading':return sw_score(text,row['reference'])
    raise ValueError(row['task'])

def behavior(text):
    t=tokens(text);counts=collections.Counter(tuple(t[i:i+4]) for i in range(max(0,len(t)-3)))
    return {'repeated_4gram_fraction':sum(n-1 for n in counts.values())/max(1,len(t)-3),'words':len(t)}
