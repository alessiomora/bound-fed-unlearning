"""Independent retained-only penultimate probe; no forget/test labels used for fitting."""
def measure(m,feature,batches,parts,split,ytrain,seed=2026,forget_groups=None):
 import numpy as np,torch
 from torch import nn
 from torch.nn import functional as F
 m.eval();rng=np.random.default_rng(2026);fit=[];ev=[]
 for c in range(1,10):
  for y in range(100):
   ix=rng.permutation(parts[c][ytrain[parts[c]]==y]);n=max(1,int(.8*len(ix))) if len(ix) else 0;fit.extend(ix[:n]);ev.extend(ix[n:])
 fit=np.asarray(fit,dtype=np.int64);ev=np.asarray(ev,dtype=np.int64)
 assert not np.intersect1d(fit,parts[0]).size
 with torch.no_grad():
  hh=[];yy=[]
  for xx,y in batches(np.arange(50000),False,None):hh.append(feature(xx));yy.append(y)
  h=torch.cat(hh);labels=torch.cat(yy);hu=[];yu=[]
  for xx,y in batches(split['unseen'],False,None,'test'):hu.append(feature(xx));yu.append(y)
  hu=torch.cat(hu);yu=torch.cat(yu);mu=h[fit].mean(0);std=h[fit].std(0).clamp_min(1e-4);z=(h-mu)/std;zu=(hu-mu)/std
 with torch.random.fork_rng(devices=[0]):
  head=nn.Linear(512,100).cuda()
 with torch.no_grad():head.weight.zero_();head.bias.zero_()
 opt=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=400,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-5,tolerance_change=1e-6)
 def closure():
  opt.zero_grad();loss=F.cross_entropy(head(z[fit]),labels[fit])+.001*head.weight.square().sum();loss.backward();return loss
 opt.step(closure)
 with torch.no_grad():
  pred=head(z).argmax(1);pu=head(zu).argmax(1);freq=torch.bincount(labels[parts[0]],minlength=100).float();freq/=freq.sum();result={}
  for name,pp,ll in [('forget',pred[parts[0]],labels[parts[0]]),('retain_eval',pred[ev],labels[ev]),('unseen',pu,yu)]:
   counts=torch.bincount(ll,minlength=100);assert (counts[freq>0]>0).all()
   w=freq[ll]/counts[ll];result[name]={'accuracy':(pp==ll).float().mean().item(),'class_matched_accuracy':(w*(pp==ll)).sum().item(),'n':len(ll)}
  if forget_groups is not None:
   result['forget_subgroups']={name:{'accuracy':(pred[ids]==labels[ids]).float().mean().item(),'n':len(ids)}
                              for name,ids in forget_groups.items()}
 nit=opt.state[head.weight]['n_iter'];result.update(iterations=nit,reached_cap=nit>=400,fit_n=len(fit),regularization=.001)
 return result
