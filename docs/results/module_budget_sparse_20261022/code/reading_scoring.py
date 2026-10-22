"""Frozen accepted reading functions, unchanged source slices; mathematics excluded."""
import re,unicodedata,collections

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


def behavior(text):
    t=tokens(text);counts=collections.Counter(tuple(t[i:i+4]) for i in range(max(0,len(t)-3)))
    return {'repeated_4gram_fraction':sum(n-1 for n in counts.values())/max(1,len(t)-3),'words':len(t)}

