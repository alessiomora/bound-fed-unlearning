"""Retain-only recovery; resumable per-round metrics, probe, cosine prototypes."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--regime',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--rounds',type=int,default=20);p.add_argument('--scope',choices=['block','encoder'],required=True);p.add_argument('--objective',choices=['feature','margin'],required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--beta',type=float,default=2.);p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,default=2026);p.add_argument('--out',type=Path,required=True);p.add_argument('--loss-mia',action='store_true');p.add_argument('--role',choices=('BOUND','Original','Retrain'),default='BOUND');a=p.parse_args()
if a.rounds<0 or a.epochs<1 or a.beta<=0 or a.lr<=0:p.error('rounds must be nonnegative and epochs, beta, and lr must be positive')
if a.role=='Original':a.rounds=0
base=Path(__file__).resolve().parents[1];out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
src=(base/'train_pair.py').read_text();setup=src[:src.index('seed_all(a.seed);m=make_model()')];start=setup.index('a=p.parse_args()');setup=setup[:start]+"a=argparse.Namespace(dataset='cifar100',regime=ARGS.regime,seed=ARGS.seed,rounds=20,out=OUT,smoke=False,pilot=False,extract=False,role='Original')\n"+setup[start+len('a=p.parse_args()\n'):]
setup=setup.replace("gpu_lock=(HERE/'gpu.lock').open('w')","gpu_lock=(HERE/'bound-ablation-gpu.lock').open('w')")
ns={'__file__':str(base/'train_pair.py'),'ARGS':a,'OUT':out};exec(setup,ns);exec(src[src.index('@torch.no_grad()\ndef evaluate'):src.index("roles=['Original'")],ns)
import torch,numpy as np
from torch.nn import functional as F
from fresh_probe import measure
torch.set_num_threads(1)
ns['seed_all'](a.seed)
m=ns['make_model']();parts=ns['parts'];batches=ns['batches'];split=ns['split'];clients=list(range(1,10));forget=parts[0];retain=np.concatenate([parts[c] for c in clients]);assert not np.intersect1d(forget,retain).size
cfg=vars(a)|{'out':str(out)}|{'source':str(a.source),'source_sha256':ns['digest'](a.source),'runner_sha256':ns['digest'](__file__),'clients':clients,'schedule':'Retain-only, sample-weighted FedAvg from round 201'}
if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
else:(out/'config.json').write_text(json.dumps(cfg,indent=2))
def feature(x):
 h=m.maxpool(m.relu(m.bn1(m.conv1(x))))
 for layer in [m.layer1,m.layer2,m.layer3,m.layer4]:h=layer(h)
 return m.avgpool(h).flatten(1)
@torch.no_grad()
def cache(ids,which):
 hs=[];ys=[]
 for x,y in batches(ids,False,None,which):hs.append(feature(x));ys.append(y)
 return torch.cat(hs),torch.cat(ys)
@torch.no_grad()
def prototype():
 # Same unseen-anchor centered cosine prototype definition as motivating analysis.
 ha,ya=cache(split['anchors'],'test');mu=ha.double().mean(0);ha=ha.double()-mu
 centers=F.normalize(torch.stack([ha[ya==c].mean(0) for c in range(100)]),dim=1)
 assert all((ya==c).any() for c in range(100))
 freq=torch.bincount(torch.as_tensor(ns['ytrain'][forget],device='cuda'),minlength=100).double();freq/=freq.sum();result={}
 for name,ids,which in [('forget',forget,'train'),('retain',retain,'train'),('unseen',split['unseen'],'test')]:
  h,y=cache(ids,which);sim=F.normalize(h.double()-mu,dim=1)@centers.T;pred=sim.argmax(1);true=sim[torch.arange(len(y)),y].clone();sim[torch.arange(len(y)),y]=-torch.inf;margin=true-sim.max(1).values;counts=torch.bincount(y,minlength=100);assert (counts[freq>0]>0).all();w=freq[y]/counts[y]
  result[name]={'margin':margin.mean().item(),'class_matched_margin':(w*margin).sum().item(),'accuracy':(pred==y).double().mean().item(),'class_matched_accuracy':(w*(pred==y)).sum().item(),'n':len(y)}
 result['forget_minus_unseen_margin']=result['forget']['class_matched_margin']-result['unseen']['class_matched_margin'];result['anchor_n']=len(ya);result['definition']='center on fixed unseen anchors; unit class means and unit centered evaluation features; true cosine minus maximum incorrect cosine; recomputed for each model/round; evaluation only'
 return result
beg=time.time()
def metrics(r,loss=None):
 m.eval();result={'round':r,'stage':'post_unlearning' if r==0 else 'recovery','elapsed':time.time()-beg,'train_loss':loss,'clients':{}}
 with torch.no_grad():
  for name,ids,which in [('forget',forget,'train'),('retain',retain,'train'),('validation',split['validation'],'test'),('test_unseen',split['unseen'],'test')]:
   correct=0
   for x,y in batches(ids,False,None,which):
    logits=m(x);assert torch.isfinite(logits).all();correct+=(logits.argmax(1)==y).sum().item()
   result['clients'][name]={'accuracy':correct/len(ids),'n':len(ids)}
 result['fresh_probe']=measure(m,feature,batches,parts,split,ns['ytrain']);result['prototype']=prototype()
 if a.loss_mia and r in (0,a.rounds):
  from loss_threshold_mia import evaluate_mia
  evaluate_mia(m,batches,forget,split,out,r)
 target=out/f'metrics_{r:02d}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(target);print(json.dumps(result),flush=True)
resume=out/'resume.pt'
if resume.exists():
 saved=torch.load(resume,map_location='cuda',weights_only=True);m.load_state_dict(saved['state']);r=saved['round']
else:
 m.load_state_dict(torch.load(a.source,map_location='cuda',weights_only=True));m.eval();r=0
 if a.role=='BOUND':
  from constrained_target import project
  h,y=cache(forget,'train')
  with torch.no_grad():
   logits=m.fc(h);correct=logits.argmax(1)==y;candidates=logits.clone();candidates[torch.arange(len(y)),y]=-torch.inf;k=candidates.argmax(1)
   m0=logits[torch.arange(len(y)),y]-logits[torch.arange(len(y)),k];d=m.fc.weight[k]-m.fc.weight[y];bdiff=m.fc.bias[y]-m.fc.bias[k];desired=-a.beta*m0
   eligible=correct.clone();target=h.clone();ii=torch.where(eligible)[0];solution,feasible,_=project(h[ii],d[ii],bdiff[ii]-desired[ii]);target[ii]=solution;eligible[ii[~feasible]]=False
   trainids=forget[eligible.cpu().numpy()];assert len(trainids)>0
   tt=target[eligible];yy=y[eligible];kk=k[eligible];mm=desired[eligible]
   targetlog=m.fc(tt);tm=targetlog[torch.arange(len(tt)),yy]-targetlog[torch.arange(len(tt)),kk]
   assert (tt>=0).all() and (tm<0).all();assert (tm-mm).abs().max()<1e-3*(1+mm.abs()).max()
   denom_feature=(tt-h[eligible]).square().sum(1).mean().clamp_min(1e-12);denom_margin=((1+a.beta)*m0[eligible]).square().mean().clamp_min(1e-12)
   frozen={key:v.detach().cpu().clone() for key,v in m.state_dict().items() if key.startswith('fc.') or (a.scope=='block' and not key.startswith('layer4.1.'))}
   (out/'design.json').write_text(json.dumps({'eligible_n':len(trainids),'original_correct':int(correct.sum()),'infeasible_n':int((correct&~eligible).sum()),'beta':a.beta,'classifier_frozen':True,'feature_initial_error':denom_feature.item(),'margin_initial_error':denom_margin.item(),'target_sha256':__import__('hashlib').sha256(tt.cpu().numpy().tobytes()).hexdigest(),'training_ids_sha256':__import__('hashlib').sha256(trainids.tobytes()).hexdigest()},indent=2))
  for name,par in m.named_parameters():par.requires_grad_(not name.startswith('fc.') and (a.scope=='encoder' or name.startswith('layer4.1.')))
  params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=a.lr)
  for epoch in range(1,a.epochs+1):
   m.eval();losses=[]
   for off,(xx,_) in enumerate(batches(trainids,False,None)):
    sl=slice(off*128,off*128+len(xx));opt.zero_grad(set_to_none=True);ff=feature(xx)
    if a.objective=='feature':loss=(ff-tt[sl]).square().sum(1).mean()/denom_feature
    else:
     ll=m.fc(ff);idx=torch.arange(len(ll),device='cuda');mar=ll[idx,yy[sl]]-ll[idx,kk[sl]];loss=(mar-mm[sl]).square().mean()/denom_margin
    assert torch.isfinite(loss);loss.backward();opt.step();losses.append(loss.item())
   with torch.no_grad():
    actual,_=cache(forget,'train');ll=m.fc(actual[eligible]);idx=torch.arange(len(ll),device='cuda');mar=ll[idx,yy]-ll[idx,kk]
    rec={'epoch':epoch,'optimized_loss_mean':float(np.mean(losses)),'normalized_feature_error':((actual[eligible]-tt).square().sum(1).mean()/denom_feature).item(),'normalized_margin_error':((mar-mm).square().mean()/denom_margin).item(),'crossing_fraction':(mar<0).float().mean().item(),'desired_margin_met_fraction':(mar<=mm).float().mean().item(),'native_forget_accuracy':(m.fc(actual).argmax(1)==y).float().mean().item(),'elapsed':time.time()-beg}
    for key,value in frozen.items():assert torch.equal(m.state_dict()[key].cpu(),value),key
   with (out/'unlearning-audit.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   print(json.dumps(rec),flush=True)
  ns['atomic_save'](ns['cpu_state'](m),out/'post_unlearning.pt')
 ns['atomic_save']({'state':ns['cpu_state'](m),'round':0},resume)
 if a.role=='BOUND':
  del frozen,h,y,logits,correct,candidates,k,m0,d,bdiff,desired,eligible,target,solution,tt,yy,kk,mm,params,opt,actual,ll,mar
for par in m.parameters():par.requires_grad_(True)
if not (out/f'metrics_{r:02d}.json').exists():metrics(r)
for r in range(r+1,a.rounds+1):
 tick=time.time();loss=ns['train_round'](m,clients,200+r);ns['atomic_save']({'state':ns['cpu_state'](m),'round':r},resume);metrics(r,loss)
 with (out/'timing.jsonl').open('a') as f:f.write(json.dumps({'round':r,'seconds':time.time()-tick})+'\n')
(out/'complete.json').write_text(json.dumps({'rounds':a.rounds,'seconds':time.time()-beg,'final_sha256':ns['digest'](resume)}))
