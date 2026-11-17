"""Metrics-only recovery sweep from the existing round-2000 paired checkpoints."""
import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from constrained_target import project

parser = argparse.ArgumentParser()
parser.add_argument("--request-size", type=int, choices=(1, 5), required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--role", choices=("BOUND", "Retrain", "Original"), default="BOUND")
parser.add_argument("--beta", type=float, default=2)
parser.add_argument("--lr", type=float, default=.0005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--scope", choices=("block", "stage", "encoder"), default="block")
parser.add_argument("--audit-crossings", action="store_true")
parser.add_argument("--loss-mia", action="store_true")
parser.add_argument("--recovery-rounds", type=int, choices=(50, 100), default=50)
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()
seed = args.seed
root = Path(__file__).resolve().parents[1]
reference = root / f"runs/cifar100_100clients_dirichlet01_s{seed}_r2000"
out = root / "runs/partial-recovery" / args.name
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    parser.error("run directory must be empty; choose a new --name")
source = (root / "train_pair.py").read_text()
start = source.index("a=p.parse_args()")
end = source.index("seed_all(a.seed);m=make_model()")
setup = source[:start] + """a=argparse.Namespace(dataset='cifar100',regime='dirichlet01',seed=SEED,
 rounds=2000,out=OUT,smoke=False,pilot=False,extract=False,role='Original')
""" + source[start + len("a=p.parse_args()\n"):end]
ns = dict(__file__=str(root / "train_pair.py"), OUT=out, SEED=seed)
exec(compile(setup, "train_pair_setup", "exec"), ns)
training = source[source.index("@torch.no_grad()\ndef evaluate"):source.index("roles=['Original'")]
assert ".998**(r-1)" in training
exec(compile(training.replace(".998**(r-1)", ".998**((r-1)*200/1000)"),
             "train_pair_fedavg", "exec"), ns)

ytrain, split, batches, device = (ns[key] for key in ("ytrain", "split", "batches", "dev"))
parts = {int(k): np.asarray(v, dtype=np.int64)
         for k, v in json.loads((reference / "partition.json").read_text()).items()}
assert len(parts) == 100 and np.array_equal(np.sort(np.concatenate(list(parts.values()))),
                                           np.arange(len(ytrain)))
ns["parts"] = parts
targets = list(range(args.request_size))
clients = sorted(set(parts) - set(targets))
design = json.loads((reference / "design.json").read_text())
assert design["requests"][str(args.request_size)]["clients"] == targets
assert ns["digest"](reference / "partition.json") == design["partition_sha256"]

# Reuse the original 100-client sampler, metrics and BOUND update.
prior = Path(__file__).with_name("partial_helpers.py").read_text()
functions = {"sampled", "features", "extract", "measure", "write_json"}
nodes = [node for node in ast.parse(prior).body
         if isinstance(node, ast.FunctionDef) and node.name in functions]
assert len(nodes) == len(functions)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_100client_helpers", "exec"))
update = prior[prior.index("def intervene("):prior.index("def recover(")]
for old, new in [
    ("k.startswith('layer4.1.')", "trainable(k)"),
    ("name.startswith('layer4.1.')", "trainable(name)"),
    ("+2*margin", "+args.beta*margin"),
    ("lr=5e-4", "lr=args.lr"),
    ("range(10)", "range(args.epochs)"),
]:
    assert old in update, old
    update = update.replace(old, new)
if args.audit_crossings:
    update = update.replace("    audits=[]", "    audits=[]\n    crossing_sets=[]")
    update = update.replace("            current=cpu_state()", """            audit["local_crossing_fraction"] = crossing_fraction(train_ids, y[eligible], other[eligible])
            crossing_sets.append((audit, train_ids, y[eligible], other[eligible]))
            current=cpu_state()""")
    update = update.replace("    model.requires_grad_(True)", """    for audit, ids, labels, competitors in crossing_sets:
        audit["aggregated_crossing_fraction"] = crossing_fraction(ids, labels, competitors)
    model.requires_grad_(True)""")
exec(compile(update, "parameterized_original_bound_update", "exec"))

@torch.no_grad()
def crossing_fraction(ids, labels, competitors):
    logits = model.fc(extract(ids))
    rows = torch.arange(len(ids), device=logits.device)
    return (logits[rows, competitors] > logits[rows, labels]).float().mean().item()


def trainable(name):
    if args.scope == "encoder":
        return not name.startswith("fc.")
    prefix = "layer4.1." if args.scope == "block" else "layer4."
    return name.startswith(prefix)

torch.set_num_threads(1)
ns["seed_all"](seed)
model = ns["make_model"]()
cpu_state = lambda: ns["cpu_state"](model)
checkpoint = reference / (f"Retrain{args.request_size}.pt" if args.role == "Retrain" else "Original.pt")
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
write_json(out / "config.json", vars(args) | dict(seed=seed, training_rounds=2000,
           clients_per_round=10, checkpoint=str(checkpoint),
           checkpoint_sha256=ns["digest"](checkpoint), save_checkpoints=False))
if args.loss_mia:
    from loss_threshold_mia import evaluate_mia

    forget = np.concatenate([parts[client] for client in targets])
    if args.role != "Retrain":
        original_audit = reference / f"request-{args.request_size}" / "loss-mia-original"
        original_audit.mkdir(parents=True, exist_ok=True)
        if not (original_audit / "mia_00.json").is_file():
            write_json(original_audit / "metrics_00.json", measure(targets, clients))
            evaluate_mia(model, batches, forget, split, original_audit, 0)
torch.cuda.synchronize()
started = time.perf_counter()
def recover(selected, round_index):
    return ns["train_round"](model, selected, round_index)

if args.role == "BOUND":
    intervene(cpu_state(), targets, out)
torch.cuda.synchronize()
write_json(out / "intervention-time.json", dict(seconds=time.perf_counter() - started))
last = 0 if args.role == "Original" else args.recovery_rounds
for r in range(last + 1):
    selected = sampled(clients, 2000 + r) if r else []
    started = time.perf_counter()
    loss = recover(selected, 2000 + r) if r else None
    torch.cuda.synchronize()
    recovery_seconds = time.perf_counter() - started if r else 0.
    metrics = measure(targets, clients)
    write_json(out / f"metrics_{r:02d}.json", metrics | dict(round=r, clients=selected,
               recovery_seconds=recovery_seconds, train_loss=loss))
    if args.loss_mia and r in (0, last):
        evaluate_mia(model, batches, forget, split, out, r)
    print(json.dumps(dict(round=r, **{k: metrics[k] for k in ("FA", "RA", "TA", "probeFA")})), flush=True)
write_json(out / "complete.json", dict(rounds=last, checkpoints_saved=False))
