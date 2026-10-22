import unittest,random
from pathlib import Path
import numpy as np
from development_scoring import score_record,summarize,instruction_score

class DevelopmentTests(unittest.TestCase):
    def test_chrf_statistics_reproduce_corpus_scoring(self):
        from sacrebleu.metrics import CHRF
        pairs=[('Habari za leo','Habari za leo.'),('Jina langu ni Sam.','Anaitwa Sam'),('','Habari')]
        records=[]
        for i,(hyp,ref) in enumerate(pairs):
            row={'reference':ref,'task':'en_to_sw'}
            records.append({**row,'scores':score_record('translation',row,hyp),'capped':False,'generated_tokens':4})
        expected=CHRF(char_order=6,word_order=2,beta=2).corpus_score([a for a,b in pairs],[[b for a,b in pairs]]).score
        self.assertAlmostEqual(summarize(records)['en_to_sw']['corpus_chrfpp'],expected)

    def test_numeric_equivalence_does_not_use_incidental_number(self):
        row={'reference':'1/2'}
        self.assertEqual(score_record('en_math',row,'I tried 100 times and got 0.5')['numeric_equivalence'],0)
        self.assertEqual(score_record('en_math',row,r'Thus \\boxed{0.5}')['numeric_equivalence'],1)
        self.assertEqual(score_record('en_math',row,'Answer: 0.5')['boxed_equivalence'],0)

    def test_swahili_f1_and_strict_option(self):
        self.assertEqual(score_record('sw_reading_kenswquad',{'reference':'Tunakoishi'},'TUNAKOISHI!')['token_f1'],1)
        self.assertEqual(score_record('en_reading',{'reference':'B'},'B')['strict_letter'],1)
        result=score_record('en_reading',{'reference':'B'},'The answer is B.')
        self.assertEqual(result['strict_letter'],0);self.assertEqual(result['explicit_letter'],1)

    def test_official_instruction_checks_are_seed_isolated(self):
        row={'id':'numeric_rng_guard','prompt':'Include cat.','instruction_id_list':['keywords:existence'],'kwargs':[{'keywords':['cat']}]}
        py=random.getstate();nps=np.random.get_state()
        self.assertEqual(instruction_score(row,'cat')['prompt_strict'],1)
        self.assertEqual(instruction_score(row,'dog')['prompt_strict'],0)
        self.assertEqual(random.getstate(),py);self.assertTrue(np.array_equal(np.random.get_state()[1],nps[1]))
    def test_cluster_bootstrap_resamples_sources_not_independent_lines(self):
        from compare_dev import paired_ci
        left=[{'id':str(i),'group_id':str(i//2),'task':'en_math','scores':{'numeric_equivalence':1.}} for i in range(4)]
        right=[{**r,'scores':{'numeric_equivalence':0.}} for r in left]
        got=paired_ci(left,right,resamples=200,seed=42)
        self.assertEqual(got['clusters'],2);self.assertEqual(got['paired95percentCI'],[100.,100.])
        wrong=list(reversed(right))
        with self.assertRaises(ValueError):paired_ci(left,wrong,resamples=20)

if __name__=='__main__':unittest.main()
