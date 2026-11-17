"""Tiny ImageNet interventions and paired 20-round recovery."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_tinyimagenet_vgg import VGG16, index_dataset, read_images

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "unlearning"))
from constrained_target import project
from loss_threshold_mia import evaluate_mia


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=json.loads, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), default=2026)
    args = parser.parse_args()
    job, out = args.job, args.out
    method = job["role"]
    if method not in ("Original", "Retrain", "BOUND"):
        parser.error("role must be Original, Retrain, or BOUND")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.60)
    seed = args.seed
    ref = args.reference
    protocol = json.loads((ref / "protocol.json").read_text())
    train, test = index_dataset(args.data)
    index = "\n".join(f"{p.relative_to(args.data)} {y}" for p, y in train + test)
    assert hashlib.sha256(index.encode()).hexdigest() == protocol["config"]["dataset_index_sha256"]
    labels = np.array([y for _, y in train])
    train_y = torch.as_tensor(labels, device="cuda")
    test_y = torch.as_tensor([y for _, y in test], device="cuda")
    parts = {i: np.array(ids) for i, ids in enumerate(protocol["partition"])}
    if args.smoke:
        parts = {i: ids[:16] for i, ids in parts.items()}
    clients = list(range(1, 10))
    forget = parts[0]
    retain = np.concatenate([parts[c] for c in clients])
    split = {k: np.array(v) for k, v in protocol["heldout"].items()}
    x = read_images(train)
    xt = read_images(test)
    pixels = torch.arange(64, device="cuda")

    def normalize(raw):
        return (raw - .5) / .25

    def augment(raw, generator):
        padded = F.pad(raw, (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
        offset = torch.randint(9, (len(raw), 2), device="cuda", generator=generator)
        cropped = padded[torch.arange(len(raw), device="cuda")[:, None, None],
                         pixels[None, :, None] + offset[:, 0, None, None],
                         pixels[None, None, :] + offset[:, 1, None, None]].permute(0, 3, 1, 2)
        flip = torch.rand(len(raw), device="cuda", generator=generator) < .5
        return normalize(torch.where(flip[:, None, None, None], cropped.flip(-1), cropped))

    class PixelBuffer:
        def __getitem__(self, ids):
            return x[ids].permute(0, 3, 1, 2).float().div_(255)

    raw_train = PixelBuffer()

    def batches(ids, training, generator, dataset="train"):
        for start in range(0, len(ids), 64 if training else 128):
            batch = ids[start:start + (64 if training else 128)]
            raw, y = (raw_train[batch], train_y[batch]) if dataset == "train" else (
                xt[batch].permute(0, 3, 1, 2).float().div_(255), test_y[batch])
            yield augment(raw, generator) if training else normalize(raw), y

    torch.manual_seed(seed)
    model = VGG16().cuda()
    path = ref / ("Retrain.pt" if method == "Retrain" else "Original.pt")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    assert checkpoint["round"] == 200 and protocol["config"]["seed"] == seed
    model.load_state_dict(checkpoint["model"])
    recovery_lr = job.get("recovery_lr", .001)

    def generator(tick, client=0):
        return torch.Generator(device="cuda").manual_seed(seed + 10000*tick + client)

    def state(cpu=False):
        return {k: v.detach().cpu().clone() if cpu else v.detach().clone()
                for k, v in model.state_dict().items()}

    def fit_client(client, tick, lr, weight_decay=.0005):
        g = generator(tick, client)
        ids = parts[client]
        order = torch.randperm(len(ids), generator=g, device="cuda").cpu().numpy()
        opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                              lr=lr, momentum=.9, weight_decay=weight_decay)
        model.train()
        for images, y in batches(ids[order], True, g):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), y)
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()

    def fedavg(selected, tick, lr):
        initial = state()
        average = {k: torch.zeros_like(v) for k, v in initial.items()}
        total = sum(len(parts[c]) for c in selected)
        for client in selected:
            model.load_state_dict(initial)
            fit_client(client, tick, lr)
            for key, value in model.state_dict().items():
                average[key].add_(value, alpha=len(parts[client])/total)
        model.load_state_dict(average)

    def train_round(_model, selected, tick):
        fedavg(selected, tick, recovery_lr)

    def recover(selected, tick):
        train_round(model, selected, tick)

    def encode(images):
        return model.encode(images)

    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("--out must be empty")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    (out/"config.json").write_text(json.dumps(dict(job=job, source_sha256=digest,
        seed=seed, recovery_lr=recovery_lr, recovery_rounds=20, checkpoint_saved=False), indent=2))
    torch.cuda.synchronize()
    start = time.perf_counter()
    audit = {}
    if method == "BOUND":
        model.eval()
        with torch.no_grad():
            h = torch.cat([encode(images) for images, _ in batches(forget, False, None)])
            y = train_y[forget]
            logits = model.fc(h)
            other = logits.scatter(1, y[:,None], -torch.inf).argmax(1)
            ids = torch.arange(len(y), device="cuda")
            margins = logits[ids, y] - logits[ids, other]
            keep = (logits.argmax(1) == y) & (margins > 0)
            direction = model.fc.weight[other] - model.fc.weight[y]
            goal = model.fc.bias[y]-model.fc.bias[other]+job["beta"]*margins
            target, feasible, _ = project(h[keep], direction[keep], goal[keep])
            positions = torch.where(keep)[0][feasible]
            targets = target[feasible]
            assert len(positions)
            selected_ids = forget[positions.cpu().numpy()]
            norm = (targets-h[positions]).square().sum(1).mean().clamp_min(1e-12)
        model.fc.requires_grad_(False)
        if job["scope"] == "block":
            for name, param in model.features.named_parameters():
                param.requires_grad_(int(name.split(".")[0]) >= 34)
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=job["lr"])
        for _ in range(job["epochs"]):
            for offset, (images, _) in enumerate(batches(selected_ids, False, None)):
                opt.zero_grad(set_to_none=True)
                target_batch = targets[offset*128:offset*128+len(images)]
                loss = (encode(images)-target_batch).square().sum(1).mean()/norm
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
        audit["target_count"] = len(positions)
        model.requires_grad_(True)
    torch.cuda.synchronize()
    (out/"intervention.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start, **audit)))
    rng = np.random.default_rng(2026)
    probe_fit = []
    for client in clients:
        ids = parts[client]
        for label in np.unique(labels[ids]):
            group = rng.permutation(ids[labels[ids] == label])
            probe_fit.extend(group[:max(1,int(.8*len(group)))])
    probe_fit = np.asarray(probe_fit)

    def extract(ids, dataset="train"):
        with torch.no_grad():
            return torch.cat([encode(images) for images, _ in batches(ids, False, None, dataset)])

    rounds = 0 if method == "Original" or args.smoke else 20
    for r in range(rounds+1):
        started = time.perf_counter()
        if r:
            recover(clients, 200+r)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started if r else 0
        model.eval()
        all_ids = np.concatenate((forget,retain))
        h = extract(all_ids)
        hu = extract(split["unseen"], "test") if not args.smoke else extract(split["unseen"][:16], "test")
        unseen = split["unseen"][:16] if args.smoke else split["unseen"]
        fit = np.searchsorted(np.sort(all_ids),probe_fit)
        fit = np.argsort(all_ids)[fit]
        z = h[fit]
        mean, std = z.mean(0), z.std(0).clamp_min(1e-4)
        z = (z-mean)/std
        probe = nn.Linear(h.shape[1],200,device="cuda")
        with torch.no_grad():
            probe.weight.zero_()
            probe.bias.zero_()
        opt = torch.optim.LBFGS(probe.parameters(), lr=1, max_iter=2 if args.smoke else 400,
                                 history_size=10,line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-5,tolerance_change=1e-6)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(probe(z),train_y[probe_fit])+.001*probe.weight.square().sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            correct = model.fc(h).argmax(1)==train_y[all_ids]
            row = dict(round=r,FA=correct[:len(forget)].float().mean().item(),
                       RA=correct[len(forget):].float().mean().item(),
                       TA=(model.fc(hu).argmax(1)==test_y[unseen]).float().mean().item(),
                       probeFA=(probe((h[:len(forget)]-mean)/std).argmax(1)==train_y[forget]).float().mean().item(),
                       recovery_seconds=seconds)
        temp = out/f"metrics_{r:02d}.tmp"
        temp.write_text(json.dumps(row,indent=2))
        temp.replace(out/f"metrics_{r:02d}.json")
        print(json.dumps(row),flush=True)
        if not args.smoke and r in (0,20):
            evaluate_mia(model,batches,forget,split,out,r)
    (out/"complete.json").write_text(json.dumps(dict(rounds=rounds, smoke=args.smoke)))


if __name__ == "__main__":
    main()
