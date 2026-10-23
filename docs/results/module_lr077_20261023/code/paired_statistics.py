"""Predeclared source-paired comparisons. No tuning and no pooled task score."""
from settings import *
import numpy as np,torch,torch.nn.functional as F,collections,csv,math
from sacrebleu.metrics import CHRF

def cluster_samples(groups):
 unique=sorted(set(groups));ix={g:i for i,g in enumerate(unique)};gi=np.array([ix[g] for g in groups]);weights=np.random.default_rng(42).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=10000);return unique,gi,weights

def mean_ci(a,b,groups,scale=100.):
 a=np.asarray(a);b=np.asarray(b);unique,gi,w=cluster_samples(groups);s=np.bincount(gi,weights=a-b,minlength=len(unique));n=np.bincount(gi,minlength=len(unique));samples=(w@s)/(w@n)*scale;ci=np.quantile(samples,[.025,.975]);return {'difference':float((a-b).mean()*scale),'low':float(ci[0]),'high':float(ci[1]),'n':len(a),'source_groups':len(unique),'paired':True}

def corpus_score(stats):return CHRF(char_order=6,word_order=2,beta=2)._compute_score_from_stats(np.asarray(stats).sum(0).tolist()).score

def chrf_ci(a,b,groups):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);unique,gi,w=cluster_samples(groups);ga=np.zeros((len(unique),a.shape[1]));gb=np.zeros_like(ga);np.add.at(ga,gi,a);np.add.at(gb,gi,b);sa=w@ga;sb=w@gb;metric=CHRF(char_order=6,word_order=2,beta=2);delta=np.array([metric._compute_score_from_stats(x.tolist()).score-metric._compute_score_from_stats(y.tolist()).score for x,y in zip(sa,sb)]);lo,hi=np.quantile(delta,[.025,.975]);return {'difference':corpus_score(a)-corpus_score(b),'low':float(lo),'high':float(hi),'n':len(a),'source_groups':len(unique),'paired':True}

