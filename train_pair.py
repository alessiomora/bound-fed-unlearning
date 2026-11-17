"""Paired full-participation FedAvg. One process, sequential clients and models."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, fcntl, hashlib, io, json, math, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18
from PIL import Image

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--dataset',choices=['cifar100','stanfordcars'],required=True)
p.add_argument('--regime',choices=['iid','dirichlet01'],required=True)
p.add_argument('--seed',type=int,default=2026)
p.add_argument('--rounds',type=int)
p.add_argument('--out',type=Path)
p.add_argument('--smoke',action='store_true')
p.add_argument('--pilot',action='store_true')
p.add_argument('--extract',action='store_true')
p.add_argument('--role',choices=['Original','Retrain','both'],default='both')
a=p.parse_args()
assert not (a.smoke and a.pilot)
gpu_lock=(HERE/'gpu.lock').open('w')
fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert torch.cuda.is_available()
data=Path(os.environ.get('BOUND_DATA', HERE/'data')).resolve(); dev='cuda'
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
rounds=a.rounds or (200 if a.dataset=='cifar100' else 100)
if a.smoke: rounds=1
name=f'{a.dataset}_{a.regime}_s{a.seed}'
out=a.out or HERE/('smoke' if a.smoke else 'pilot' if a.pilot else 'runs')/name
out.mkdir(parents=True,exist_ok=True)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def log(**kw):
    kw['time']=time.time(); line=json.dumps(kw)
    print(line,flush=True)
    with (out/'progress.jsonl').open('a') as f:f.write(line+'\n')
def atomic_save(obj,path):
    if shutil.disk_usage(out).free<2_000_000_000:raise RuntimeError('Less than 2 GB disk reserve')
    tmp=Path(str(path)+'.tmp');torch.save(obj,tmp);tmp.replace(path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def state(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}
def cpu_state(m):return {k:v.detach().cpu() for k,v in m.state_dict().items()}

partfile=HERE/'prepared'/a.dataset/(a.regime+'.json')
parts={int(k):np.asarray(v,dtype=np.int64) for k,v in json.loads(partfile.read_text()).items()}
labels=np.load(HERE/'prepared'/(a.dataset+'-labels.npz'))
ytrain=labels['train'];ytest=labels['test'];classes=int(ytrain.max())+1
if a.smoke:parts={k:v[:32] for k,v in parts.items()}
flat=np.concatenate(list(parts.values()));assert len(flat)==len(np.unique(flat))
# A single fixed validation/anchor/evaluation split, independent of training seed.
splitfile=HERE/'prepared'/a.dataset/'test-split.npz'
if not splitfile.exists():
    rng=np.random.default_rng(2026)
    if a.dataset=='cifar100':
        ix=rng.permutation(len(ytest));splits={'validation':ix[:2000],'anchors':ix[2000:6000],'unseen':ix[6000:]}
    else:
        splits={k:[] for k in ['validation','anchors','unseen']}
        for c in range(classes):
            ix=rng.permutation(np.flatnonzero(ytest==c));nv=round(.2*len(ix));na=(len(ix)-nv)//2
            for k,v in zip(splits,[ix[:nv],ix[nv:nv+na],ix[nv+na:]]):splits[k].extend(v)
        splits={k:np.asarray(v,dtype=np.int64) for k,v in splits.items()}
    np.savez(splitfile,**splits)
split=dict(np.load(splitfile));assert len(np.unique(np.concatenate(list(split.values()))))==len(ytest)
val=split['validation'][:32] if a.smoke else split['validation']

if a.dataset=='cifar100':
    train=CIFAR100(data,train=True,download=False);test=CIFAR100(data,train=False,download=False)
    assert np.array_equal(train.targets,ytrain) and np.array_equal(test.targets,ytest)
    x=torch.tensor(train.data,device=dev).permute(0,3,1,2).float()/255
    xt=torch.tensor(test.data,device=dev).permute(0,3,1,2).float()/255
    train_targets_tensor=torch.tensor(ytrain,device=dev);yt=torch.tensor(ytest,device=dev)
    mean=torch.tensor([.5071,.4867,.4408],device=dev)[None,:,None,None]
    std=torch.tensor([.2675,.2565,.2761],device=dev)[None,:,None,None]
    def norm(z):return (z-mean)/std
    def make_model():
        m=resnet18(weights=None,num_classes=100,norm_layer=lambda c:nn.GroupNorm(2,c,eps=1e-5))
        m.conv1=nn.Conv2d(3,64,3,padding=1,bias=False);m.maxpool=nn.Identity()
        return m.to(dev)
    def augment(z,g):
        b=len(z);z=F.pad(z,(4,4,4,4),mode='reflect')
        offsets=torch.randint(9,(b,2),generator=g,device=dev)
        rows=offsets[:,0,None,None]+torch.arange(32,device=dev)[None,:,None]
        cols=offsets[:,1,None,None]+torch.arange(32,device=dev)[None,None,:]
        z=z.permute(0,2,3,1)[torch.arange(b,device=dev)[:,None,None],rows,cols].permute(0,3,1,2)
        flip=torch.rand(b,generator=g,device=dev)<.5
        return norm(torch.where(flip[:,None,None,None],z.flip(-1),z))
    def batches(ids,training,g,which='train'):
        for start in range(0,len(ids),32 if training else 128):
            ii=ids[start:start+(32 if training else 128)]
            xx=x if which=='train' else xt;ys=train_targets_tensor if which=='train' else yt
            yield augment(xx[ii],g) if training else norm(xx[ii]),ys[ii]
else:
    import timm
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    manifest=json.loads((HERE/'prepared/suite.json').read_text())
    spec=manifest['cars_backbone'];repo=spec['model'];revision=spec['revision']
    weightpath=os.environ.get('BOUND_PRETRAINED_WEIGHT') or hf_hub_download(repo_id=repo,filename='pytorch_model.bin',revision=revision)
    weights=torch.load(weightpath,map_location='cpu',weights_only=True)
    # Record exact downloaded bytes, and never download from an unpinned revision.
    (out/'pretrained.json').write_text(json.dumps({'repo':repo,'revision':revision,'sha256':digest(weightpath)},indent=2))
    def make_model():
        m=timm.create_model('deit_tiny_patch16_224',pretrained=False,num_classes=1000)
        m.load_state_dict(weights,strict=True);m.reset_classifier(classes)
        return m.to(dev)
    image_hashes=json.loads((HERE/'prepared/stanfordcars/image-hashes.json').read_text())
    records={}
    for which in ['train','test']:
        images={}
        for path in sorted((data/'stanford-cars-parquet').glob(which+'-*.parquet')):
            table=pq.read_table(path,columns=['image','label'])
            for row in table.to_pylist():
                raw=row['image']['bytes']
                images[hashlib.sha256(raw).hexdigest()]=(raw,row['label'])
        missing=set(image_hashes[which])-images.keys()
        if missing:
            raise ValueError(f'{which}: {len(missing)} reference images missing from Cars shards')
        ordered=[images[h] for h in image_hashes[which]]
        records[which]=[raw for raw,_ in ordered]
        assert np.array_equal([label for _,label in ordered],ytrain if which=='train' else ytest)
    cfg=timm.data.resolve_model_data_config(make_model())
    eval_transform=timm.data.create_transform(**cfg,is_training=False)
    # Explicit conventional fine-tuning transform; no Mixup/RandAugment/erasing.
    train_transform=T.Compose([T.RandomResizedCrop(224,interpolation=T.InterpolationMode.BICUBIC),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(cfg['mean'],cfg['std'])])
    (out/'transforms.json').write_text(json.dumps({'checkpoint_eval':cfg,'train':'RandomResizedCrop(224,scale=(0.08,1),ratio=(3/4,4/3),bicubic),RandomHorizontalFlip(0.5),ToTensor,checkpoint Normalize','amp':False},indent=2))
    def batches(ids,training,g,which='train'):
        bs=16 if training else 64
        for start in range(0,len(ids),bs):
            ii=ids[start:start+bs];tr=train_transform if training else eval_transform
            images=[]
            for i in ii:
                with Image.open(io.BytesIO(records[which][int(i)])) as im:images.append(tr(im.convert('RGB')))
            yield torch.stack(images).to(dev),torch.as_tensor((ytrain if which=='train' else ytest)[ii],device=dev)

seed_all(a.seed);m=make_model()
initial=cpu_state(m);initial_path=out/'Initial.pt'
if initial_path.exists():
    loaded=torch.load(initial_path,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,loaded[k]) for k,v in initial.items()),'Initial-state mismatch'
else:atomic_save(initial,initial_path)
config=vars(a).copy();config['out']=str(out);config.update(rounds=rounds,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),initial_sha256=digest(initial_path),partition_sha256=digest(partfile),test_split_sha256=digest(splitfile),source_sha256=digest(__file__),precision='float32, TF32 disabled',clients={k:len(v) for k,v in parts.items()},paired_rng='seed+10000*round+client; restart all CPU/CUDA/Python RNGs per client; CIFAR augment uses explicit CUDA generator')
# Role and feature-extraction flags do not change the training protocol.
configfile=out/'config.json'
if configfile.exists():
    old=json.loads(configfile.read_text())
    for k in ['rounds','seed','dataset','regime','initial_sha256','partition_sha256','source_sha256']:
        assert old[k]==config[k],f'Configuration mismatch: {k}'
else:configfile.write_text(json.dumps(config,indent=2))

@torch.no_grad()
def evaluate(m,ids):
    m.eval();correct=0;loss=0.
    for xx,yy in batches(ids,False,None,'test'):
        zz=m(xx);assert torch.isfinite(zz).all()
        correct+=(zz.argmax(1)==yy).sum().item();loss+=F.cross_entropy(zz,yy,reduction='sum').item()
    return {'accuracy':correct/len(ids),'loss':loss/len(ids),'n':len(ids)}

def train_round(m,clients,r):
    base=state(m);agg={k:torch.zeros_like(v) for k,v in base.items()};total=sum(len(parts[c]) for c in clients)
    losses=[]
    for c in clients:
        s=a.seed+10000*r+c;seed_all(s)
        g=torch.Generator(device=dev).manual_seed(s)
        m.load_state_dict(base);m.train()
        if a.dataset=='cifar100':opt=torch.optim.SGD(m.parameters(),lr=.1*.998**(r-1),momentum=0,weight_decay=0)
        else:
            lr=5e-5*.5*(1+math.cos(math.pi*(r-1)/100))
            opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.05)
        ids=torch.as_tensor(parts[c],device=dev);ids=ids[torch.randperm(len(ids),generator=g,device=dev)].cpu().numpy()
        for xx,yy in batches(ids,True,g):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(xx),yy)
            assert torch.isfinite(loss),'Nonfinite loss';loss.backward();opt.step();losses.append(loss.item())
        for k,v in m.state_dict().items():agg[k].add_(v,alpha=len(ids)/total)
    m.load_state_dict(agg)
    return float(np.mean(losses))

roles=['Original','Retrain'] if a.role=='both' else [a.role]
for role in roles:
    if a.extract:break
    final=out/(role+'.pt');done=out/(role+'-complete.json')
    if done.exists():
        assert final.exists() and json.loads(done.read_text())['sha256']==digest(final)
        log(phase='already-complete',role=role);continue
    clients=list(parts) if role=='Original' else [c for c in parts if c!=0]
    resume=out/(role+'-resume.pt');m.load_state_dict(initial);first=1
    if resume.exists():
        saved=torch.load(resume,map_location=dev,weights_only=True);m.load_state_dict(saved['state']);first=saved['round']+1
    log(phase='start',role=role,start_round=first,rounds=rounds,validation=evaluate(m,val))
    for r in range(first,rounds+1):
        begin=time.time();loss=train_round(m,clients,r)
        validation=evaluate(m,val)
        atomic_save({'state':cpu_state(m),'round':r},resume)
        log(phase='round',role=role,round=r,seconds=time.time()-begin,train_loss=loss,validation=validation,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
    atomic_save(cpu_state(m),final)
    done.write_text(json.dumps({'rounds':rounds,'sha256':digest(final),'validation':evaluate(m,val)},indent=2))
    resume.unlink(missing_ok=True)
    log(phase='complete',role=role,sha256=digest(final))

if a.extract:
    # Four pooled feature stages; no full maps or token tensors are saved.
    eval_ids=np.concatenate([split['anchors'],split['unseen']]);train_ids=np.sort(flat)
    np.savez(out/'feature-indices.npz',train_ids=train_ids,test_ids=eval_ids,labels=np.concatenate([ytrain[train_ids],ytest[eval_ids]]),anchors=len(split['anchors']),forget=np.isin(train_ids,parts[0]))
    for role in roles:
        m.load_state_dict(torch.load(out/(role+'.pt'),map_location=dev,weights_only=True));m.eval()
        dims=[64,128,256,512] if a.dataset=='cifar100' else [192]*4
        feature_bytes=(len(train_ids)+len(eval_ids))*sum(dims)*4
        assert shutil.disk_usage(out).free>2_000_000_000+feature_bytes,'Insufficient disk reserve for pooled features'
        maps=[np.lib.format.open_memmap(out/f'{role}_layer{i+1}.npy',mode='w+',dtype=np.float32,shape=(len(train_ids)+len(eval_ids),d)) for i,d in enumerate(dims)]
        captures={};handles=[]
        if a.dataset=='cifar100':
            for i,layer in enumerate([m.layer1,m.layer2,m.layer3,m.layer4]):
                handles.append(layer.register_forward_hook(lambda mod,ins,outs,i=i:captures.__setitem__(i,outs.mean((2,3)))))
        else:
            for i,j in enumerate([2,5,8]):
                handles.append(m.blocks[j].register_forward_hook(lambda mod,ins,outs,i=i:captures.__setitem__(i,outs[:,0])))
            handles.append(m.norm.register_forward_hook(lambda mod,ins,outs:captures.__setitem__(3,outs[:,0])))
        offset=0
        with torch.no_grad():
            for which,ids in [('train',train_ids),('test',eval_ids)]:
                for xx,yy in batches(ids,False,None,which):
                    m(xx)
                    for i,arr in enumerate(maps):arr[offset:offset+len(xx)]=captures[i].float().cpu().numpy()
                    offset+=len(xx)
        for h in handles:h.remove()
        for arr in maps:arr.flush()
        assert offset==len(train_ids)+len(eval_ids)
        log(phase='features-complete',role=role,n=offset,dims=dims)
