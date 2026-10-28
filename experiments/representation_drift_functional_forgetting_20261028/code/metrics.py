"""FP64 representation metrics; covariance is over samples at one fixed layer."""
import numpy as np

def paired_metrics(base,adapted):
 b=np.asarray(base,dtype=np.float64);a=np.asarray(adapted,dtype=np.float64)
 assert b.shape==a.shape and np.isfinite(b).all() and np.isfinite(a).all()
 bn=np.linalg.norm(b,axis=-1);an=np.linalg.norm(a,axis=-1);d=a-b
 return dict(cosine_drift=1-np.clip(np.sum(b*a,axis=-1)/np.maximum(bn*an,1e-12),-1,1),
  normalized_l2=np.linalg.norm(d,axis=-1)/np.maximum(bn,1e-12),
  base_norm=bn,adapted_norm=an,norm_ratio=an/np.maximum(bn,1e-12),
  near_zero_norm=(bn<1e-12)|(an<1e-12))

def covariance_drift(base,adapted):
 b=np.asarray(base,dtype=np.float64);a=np.asarray(adapted,dtype=np.float64)
 assert a.shape==b.shape and b.ndim==2 and len(b)>1
 b=b-b.mean(0);a=a-a.mean(0);den=(len(b)-1)**2
 # ||C_b||F^2=||b b^T||F^2/(N-1)^2 and <C_b,C_a>=||b a^T||F^2/(N-1)^2.
 bb=b@b.T;nb=float(np.square(bb).sum()/den);na=float(np.square(a@a.T).sum()/den)
 # Expand the covariance difference in d=a-b, avoiding subtraction of two O(1) covariances.
 d=a-b;dd=d@d.T;bd=b@d.T
 terms=np.array([2*np.sum(bb*dd),2*np.sum(bd*bd.T),4*np.sum(bd*dd),np.square(dd).sum()])/den
 delta=float(terms.sum())
 assert delta>=-1e-12*max(float(np.abs(terms).sum()),1e-300)
 delta=max(0.,delta)
 return dict(relative_frobenius=float(np.sqrt(delta)/max(np.sqrt(nb),1e-12)),
  absolute_frobenius=float(np.sqrt(delta)),base_frobenius=float(np.sqrt(nb)),
  adapted_frobenius=float(np.sqrt(na)),samples=len(b),features=b.shape[1],
  definition='sample covariance; centered over samples; denominator N-1; FP64 Gram identity')

def functional_state(base,adapted):
 if base is None or adapted is None:return 'Unresolved'
 return ('Retained' if adapted else 'Forgotten') if base else ('Acquired' if adapted else 'Still Wrong')

def group_mean_intervals(values,groups,repeats=2000,seed=42):
 x=np.asarray(values,dtype=np.float64);assert len(x)==len(groups)
 if x.ndim==1:x=x[:,None]
 keys=list(dict.fromkeys(groups));index={g:i for i,g in enumerate(keys)}
 sums=np.zeros((len(keys),x.shape[1]));counts=np.zeros(len(keys))
 for row,g in zip(x,groups):sums[index[g]]+=row;counts[index[g]]+=1
 rng=np.random.default_rng(seed);weights=rng.multinomial(len(keys),np.ones(len(keys))/len(keys),size=repeats)
 scores=(weights@sums)/(weights@counts)[:,None]
 return np.mean(x,axis=0),np.quantile(scores,[.025,.975],axis=0),len(keys)

def covariance_spectral_drift(base,adapted):
 """Exact low-rank matrix-2-norm, separately named from Frobenius drift."""
 b=np.asarray(base,dtype=np.float64);a=np.asarray(adapted,dtype=np.float64)
 assert a.shape==b.shape and b.ndim==2 and len(b)>1
 b=b-b.mean(0);a=a-a.mean(0);d=a-b;n=len(b)
 # Delta C = b.T d + d.T b + d.T d, projected to its at-most-2N row span.
 _,r=np.linalg.qr(np.concatenate((b,d),axis=0).T,mode='reduced')
 rb,rd=r[:,:n],r[:,n:]
 delta=(rb@rd.T+rd@rb.T+rd@rd.T)/(n-1)
 delta=(delta+delta.T)/2
 absolute=float(np.abs(np.linalg.eigvalsh(delta)).max())
 base_spectral=float(np.linalg.eigvalsh(b@b.T).max()/(n-1))
 return dict(absolute_matrix_spectral_norm_2=absolute,base_matrix_spectral_norm_2=base_spectral,
  relative_matrix_spectral_norm_2=absolute/max(base_spectral,1e-12))
