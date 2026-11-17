"""Evaluate Cars references or run BOUND with retain-only recovery."""
import argparse
from pathlib import Path
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from loss_threshold_mia import evaluate_mia

parser = argparse.ArgumentParser()
parser.add_argument('--regime', choices=('iid','dirichlet01'), required=True)
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--role', choices=('Original','Retrain','BOUND'), default='BOUND')
parser.add_argument('--source', type=Path, required=True, help='Original.pt or Retrain.pt')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--beta', type=float, default=2)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--scope', choices=('full_encoder','final_block'), default='full_encoder')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--rounds', type=int, default=10)
args = parser.parse_args()
if args.out.exists() and any(args.out.iterdir()):
    parser.error('--out must be empty')
runner = Path(__file__).resolve().parents[1] / 'train_pair.py'
source_code = runner.read_text()
setup = source_code.split('seed_all(a.seed);m=make_model()',1)[0]
setup = setup.replace('a=p.parse_args()', """a=argparse.Namespace(dataset='stanfordcars',regime=args.regime,seed=args.seed,
rounds=100,out=args.out,smoke=False,pilot=False,extract=False,role='Original')""")
__file__ = str(runner)
exec(compile(setup, str(runner), 'exec'))
seed_all(a.seed)
m = make_model()
clients = sorted(c for c in parts if c != 0)

forget = parts[0]
retain = np.concatenate([ids for client, ids in parts.items() if client != 0])
unseen = split['unseen']
rng = np.random.default_rng(2026)
fit = np.concatenate([
    rng.permutation(ids[ytrain[ids] == label])[:max(1, int(.8 * np.sum(ytrain[ids] == label)))]
    for client, ids in parts.items() if client != 0
    for label in np.unique(ytrain[ids])
])

def features(images):
    return m.forward_head(m.forward_features(images), pre_logits=True)

@torch.no_grad()
def extract(ids, which='train'):
    m.eval()
    return torch.cat([features(images) for images, _ in batches(ids, False, None, which)])

def measure(role, round_number):
    train = extract(np.concatenate([forget, retain]))
    test = extract(unseen, 'test')
    labels = torch.as_tensor(ytrain[np.concatenate([forget, retain])], device=dev)
    fit_indices = torch.as_tensor(np.flatnonzero(np.isin(retain, fit)) + len(forget), device=dev)
    fit_features = train[fit_indices]
    mean = fit_features.mean(0)
    std = fit_features.std(0).clamp_min(1e-4)
    normalized = (fit_features - mean) / std
    probe = nn.Linear(train.shape[1], classes).to(dev)
    with torch.no_grad():
        probe.weight.zero_()
        probe.bias.zero_()
    opt = torch.optim.LBFGS(probe.parameters(), max_iter=400, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(probe(normalized), labels[fit_indices]) + .001 * probe.weight.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        n = len(forget)
        pred = m.head(train).argmax(1)
        log(phase='evaluation', role=role, round=round_number,
            FA=(pred[:n] == labels[:n]).float().mean().item(),
            RA=(pred[n:] == labels[n:]).float().mean().item(),
            TA=(m.head(test).argmax(1) == torch.as_tensor(ytest[unseen], device=dev)).float().mean().item(),
            probeFA=(probe((train[:n]-mean)/std).argmax(1) == labels[:n]).float().mean().item())


def local_fit(c, tick, lr, optimizer='adam', loss_fn=F.cross_entropy, epochs=1):
    seed = a.seed+10000*tick+c
    seed_all(seed)
    generator = torch.Generator(device=dev).manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    if optimizer=='sgd':
        opt=torch.optim.SGD(params,lr=lr,momentum=.9,weight_decay=5e-4)
    else:
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=.05)
    m.train()
    ids=parts[c]
    for _ in range(epochs):
        order=torch.randperm(len(ids),generator=generator,device=dev).cpu().numpy()
        for x,y in batches(ids[order],True,generator):
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(m(x),y)
            assert torch.isfinite(loss), 'nonfinite local loss'
            loss.backward()
            opt.step()

def vector():
    return parameters_to_vector(m.parameters()).detach()

def set_vector(v):
    with torch.no_grad():
        vector_to_parameters(v.clone(),m.parameters())

def local_delta(c,tick,lr,base,loss_fn=F.cross_entropy):
    set_vector(base)
    local_fit(c,tick,lr,loss_fn=loss_fn)
    return vector()-base

def recovery(r):
    base=vector()
    lr=1e-5
    total=sum(len(parts[c]) for c in clients)
    update=torch.zeros_like(base)
    for c in clients:
        delta=local_delta(c,100+r,lr,base)
        update.add_(delta,alpha=len(parts[c])/total)
    set_vector(base+update)

m.load_state_dict(torch.load(args.source,map_location=dev,weights_only=True))
log(phase='design', role=args.role, source_sha256=digest(args.source), recovery_lr=1e-5)
if args.role == 'BOUND':
    beta, lr, scope = args.beta, args.lr, args.scope
    z = extract(forget)
    y = torch.as_tensor(ytrain[forget], device=dev)
    with torch.no_grad():
        logits = m.head(z)
        competitors = logits.clone()
        competitors.scatter_(1, y[:, None], -torch.inf)
        k = competitors.argmax(1)
        margin = logits.gather(1, y[:, None]).flatten() - logits.gather(1, k[:, None]).flatten()
        eligible = (logits.argmax(1) == y) & (margin > 0)
        direction = m.head.weight[k] - m.head.weight[y]
        norm2 = direction.square().sum(1)
        eligible &= norm2 > 0
        targets = z[eligible] + ((1 + beta) * margin[eligible] / norm2[eligible])[:, None] * direction[eligible]
        normalizer = (targets-z[eligible]).square().sum(1).mean()
        assert torch.isfinite(normalizer) and normalizer > 0

    ids = forget[eligible.cpu().numpy()]
    head = {key: value.clone() for key, value in m.head.state_dict().items()}
    m.requires_grad_(False)
    if scope == 'final_block':
        m.blocks[-1].requires_grad_(True)
    else:
        assert scope == 'full_encoder'
        m.requires_grad_(True)
        m.head.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in m.parameters() if p.requires_grad), lr=lr)
    log(phase='design', beta=beta, lr=lr, scope=scope, epochs=args.epochs, eligible=len(ids), forget=len(forget))
    for epoch in range(1, args.epochs + 1):
        offset = 0
        for images, _ in batches(ids, False, None):
            optimizer.zero_grad()
            loss = (features(images)-targets[offset:offset+len(images)]).square().sum(1).mean()/normalizer
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            offset += len(images)
        log(phase='unlearning', epoch=epoch, feature_error=((extract(ids)-targets).square().sum(1).mean()/normalizer).item())
    assert all(torch.equal(value, m.head.state_dict()[key]) for key, value in head.items())
    with torch.no_grad():
        logits = m.head(extract(forget))
        crossing = (logits.gather(1,k[:,None]).flatten() > logits.gather(1,y[:,None]).flatten())
        log(phase='crossing', fraction=crossing[eligible].float().mean().item())
    m.requires_grad_(True)
last = 0 if args.role == 'Original' else args.rounds
for r in range(last+1):
    if r:
        recovery(r)
    measure(args.role,r)
    if r in (0,last):
        evaluate_mia(m,batches,forget,split,out,r)
(out/'complete.json').write_text(json.dumps({'rounds':last,'role':args.role}))
