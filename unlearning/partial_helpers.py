def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(path)

def sampled(clients,r):
    # Filter a common random ordering, pairing schedules wherever possible.
    rng=np.random.default_rng(np.random.SeedSequence([seed,901,r]))
    allowed=set(clients)
    return [int(c) for c in rng.permutation(100) if c in allowed][:10]

def features(x):
    h=model.maxpool(model.relu(model.bn1(model.conv1(x))))
    for layer in (model.layer1,model.layer2,model.layer3,model.layer4):
        h=layer(h)
    return model.avgpool(h).flatten(1)

def extract(ids,which='train'):
    model.eval()
    return torch.cat([features(x) for x,_ in batches(ids,False,None,which)])

def measure(targets,clients):
    retain=np.concatenate([parts[c] for c in clients])
    forget=np.concatenate([parts[c] for c in targets])
    h=extract(np.arange(len(ytrain)))
    labels=torch.as_tensor(ytrain,device=device)
    rng=np.random.default_rng(seed)
    fit=[]
    for c in clients:
        ids=parts[c]
        for label in np.unique(ytrain[ids]):
            group=rng.permutation(ids[ytrain[ids]==label])
            fit.extend(group[:max(1,int(.8*len(group)))])
    assert not np.intersect1d(fit,forget).size
    fit=torch.as_tensor(fit,device=device)
    mean=h[fit].mean(0)
    std=h[fit].std(0).clamp_min(1e-4)
    z=(h-mean)/std
    probe=torch.nn.Linear(h.shape[1],100,device=device)
    with torch.no_grad():
        probe.weight.zero_()
        probe.bias.zero_()
    opt=torch.optim.LBFGS(probe.parameters(),lr=1,max_iter=400,history_size=10,
                          tolerance_grad=1e-5,tolerance_change=1e-6,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss=F.cross_entropy(probe(z[fit]),labels[fit])+.001*probe.weight.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        correct=model.fc(h).argmax(1)==labels
        probe_correct=probe(z).argmax(1)==labels
        test=ns['evaluate'](model,split['unseen'])['accuracy']
        return dict(FA=correct[forget].float().mean().item(),RA=correct[retain].float().mean().item(),TA=test,
                    probeFA=probe_correct[forget].float().mean().item(),
                    per_client={c:dict(n=len(parts[c]),FA=correct[parts[c]].float().mean().item(),
                        probeFA=probe_correct[parts[c]].float().mean().item()) for c in targets})

def intervene(original,targets,folder):
    aggregate={k:torch.zeros_like(v) for k,v in original.items() if k.startswith('layer4.1.')}
    total=0
    audits=[]
    for c in targets:
        model.load_state_dict(original)
        ids=parts[c]
        h=extract(ids)
        y=torch.as_tensor(ytrain[ids],device=device)
        with torch.no_grad():
            logits=model.fc(h)
            correct=logits.argmax(1)==y
            other=logits.scatter(1,y[:,None],-torch.inf).argmax(1)
            i=torch.arange(len(y),device=device)
            margin=logits[i,y]-logits[i,other]
            eligible=torch.where(correct & (margin>0))[0]
            d=model.fc.weight[other]-model.fc.weight[y]
            goal=model.fc.bias[y]-model.fc.bias[other]+2*margin
            wanted,feasible,_=project(h[eligible],d[eligible],goal[eligible])
            wanted=wanted[feasible]
            eligible=eligible[feasible]
        n=len(eligible)
        audit=dict(client=c,examples=len(ids),correct=int(correct.sum()),eligible=n)
        if n:
            train_ids=ids[eligible.cpu().numpy()]
            norm=(wanted-h[eligible]).square().sum(1).mean().clamp_min(1e-12)
            for name,param in model.named_parameters():
                param.requires_grad_(name.startswith('layer4.1.'))
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-4)
            for epoch in range(10):
                offset=0
                for x,_ in batches(train_ids,False,None):
                    opt.zero_grad(set_to_none=True)
                    loss=(features(x)-wanted[offset:offset+len(x)]).square().sum(1).mean()/norm
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    offset+=len(x)
            current=cpu_state()
            assert all(torch.equal(v,original[k]) for k,v in current.items() if k not in aggregate)
            for k in aggregate:
                aggregate[k].add_(current[k]-original[k],alpha=n)
            total+=n
        audits.append(audit)
    assert total>0,'No eligible examples across requesting clients'
    model.load_state_dict({k:v+aggregate[k]/total if k in aggregate else v for k,v in original.items()})
    model.requires_grad_(True)
    write_json(folder/'aggregation.json',dict(total_eligible=total,clients=audits))
