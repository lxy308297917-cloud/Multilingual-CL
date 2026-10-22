"""CPU descriptive geometry of the actual fixed2000 target bank; no new objective."""
import torch
import torch.nn.functional as F
from settings import *
torch.set_num_threads(4)
obj=torch.load(TRAIN/'EN_TEACHER.pt',map_location='cpu',weights_only=False);x=obj['vectors'].float();z=F.normalize(x,dim=-1);sim=z@z.T;n=len(z);mask=~torch.eye(n,dtype=torch.bool);off=sim[mask];mean=z.mean(0)
report={'teacher_sha256':sha(TRAIN/'EN_TEACHER.pt'),'rows':n,'dimensions':x.shape[1],'layer':obj['layer'],'raw_norm_quantiles':torch.quantile(x.norm(dim=1),torch.tensor([0.,.1,.5,.9,1.])).tolist(),'normalized_target_mean_norm_squared':float(mean.square().sum()),'off_diagonal_cosine_quantiles':torch.quantile(off,torch.tensor([0.,.1,.5,.9,1.])).tolist(),'off_diagonal_cosine_mean':float(off.mean()),'normalized_per_dimension_variance_mean':float(z.var(0).mean()),'normalized_variance_trace':float(z.var(0).sum()),'candidate_uniform_cross_entropy_log2000':float(torch.tensor(float(n)).log()),'temperature':.07,'interpretation':'Descriptive common-direction dominance of original uncentered mean-pooled target vectors. Does not establish that centering or changing pooling improves tasks.','no_new_training_or_layer_selection':True}
write(REPORT/'FIXED_BANK_GEOMETRY.json',report);print(json.dumps(report))
