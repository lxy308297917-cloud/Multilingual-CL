"""Numerical identities only; these fixtures are never scientific results."""
from pathlib import Path
import numpy as np,json,time
from metrics import covariance_drift,covariance_spectral_drift,paired_metrics,functional_state
rng=np.random.default_rng(42);b=rng.normal(size=(19,37));a=b+.2*rng.normal(size=b.shape)
cb=np.cov(b,rowvar=False,ddof=1);ca=np.cov(a,rowvar=False,ddof=1)
expected=np.linalg.norm(ca-cb)/np.linalg.norm(cb);actual=covariance_drift(b,a)['relative_frobenius']
assert np.isclose(expected,actual,rtol=1e-12,atol=1e-12)
tiny=b+1e-10*rng.normal(size=b.shape)
tiny_expected=np.linalg.norm(np.cov(tiny,rowvar=False,ddof=1)-cb)/np.linalg.norm(cb)
tiny_actual=covariance_drift(b,tiny)['relative_frobenius']
assert np.isclose(tiny_expected,tiny_actual,rtol=1e-5,atol=1e-15),(tiny_expected,tiny_actual)
spectral=covariance_spectral_drift(b,a)
expected_spectral=float(np.abs(np.linalg.eigvalsh(ca-cb)).max())
assert np.isclose(spectral['absolute_matrix_spectral_norm_2'],expected_spectral,rtol=1e-12,atol=1e-12)
m=paired_metrics(b,b);assert np.max(np.abs(m['cosine_drift']))<1e-14 and not m['normalized_l2'].any()
assert covariance_drift(b,b)['relative_frobenius']<1e-7
assert [functional_state(x,y) for x,y in [(True,True),(True,False),(False,True),(False,False),(True,None)]]==['Retained','Forgotten','Acquired','Still Wrong','Unresolved']
p=Path(__file__).resolve().parents[3]/'docs/results/representation_drift_functional_forgetting_20261028/NUMERICAL_ACCEPTANCE.json'
p.write_text(json.dumps(dict(passed=True,covariance_gram_matches_explicit_covariance=True,
relative_error=abs(actual-expected)/expected,identity_and_state_mapping_checked=True,tiny_covariance_shift_preserved=True,matrix_spectral_norm_matches_explicit_eigenvalues=True,
scope='Small CPU identity check only; no synthetic scientific figures or results',unix=time.time()),indent=2)+'\n')
print(p)
