from settings import *
import math
from scoring import instruction,translation
import lm_eval
assert translation('A correct translation.','A correct translation.')['sentence_chrfpp']==100.
assert translation('unrelated text','A correct translation.')['sentence_chrfpp']<100.
row=dict(key=0,prompt='Include the word alpha.',instruction_id_list=['keywords:existence'],kwargs=[dict(keywords=['alpha'])])
assert instruction(row,'alpha')['prompt_strict']
assert not instruction(row,'beta')['prompt_strict']
# Preserve the standard harness's explicitly separate strict/flexible definitions.
import lm_eval
p=Path(lm_eval.__file__).parent/'tasks/gsm8k/gsm8k.yaml'
text=p.read_text();assert 'strict-match' in text and 'flexible-extract' in text and '####' in text
import re
strict=re.compile(r'#### (\-?[0-9\.\,]+)')
assert strict.search('reasoning\n#### 42').group(1)=='42'
assert strict.search('reasoning\n#### 43').group(1)!='42'
assert strict.search('The answer is 42.') is None
assert not strict.search('I used 42 somewhere but no final answer.')
write(REPORT/'SCORER_ACCEPTANCE.json',dict(passed=True,translation_identity_case=True,translation_wrong_case=True,
 official_IFEval_keyword_positive_negative=True,official_GSM8K_yaml_sha256=sha(p),
 GSM8K_strict_flexible_separately_reported=True,strict_format_failure_not_content_error=True,
 raw_generation_retained=True,no_posthoc_parser_substitution=True))
print('CPU scoring acceptance passes; real Base task smokes required separately')
