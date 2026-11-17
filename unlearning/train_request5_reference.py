"""Train paired 100-client Original and retained-only Retrain on a fixed partition."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--seed", required=True, type=int, choices=(2026, 2027, 2028))
p.add_argument("--check-only", action="store_true")
p.add_argument("--request-size", type=int, choices=(1, 5), default=5)
args = p.parse_args()
seed = args.seed
request = str(args.request_size)
root = Path(__file__).resolve().parents[1]
reference = root / "prepared/cifar100-100clients"
out = root / f"runs/cifar100_100clients_dirichlet01_s{seed}_r2000"
source = (root / "train_pair.py").read_text()
round_function, = (
    node for node in ast.parse(source).body
    if isinstance(node, ast.FunctionDef) and node.name == "train_round"
)
training = ast.get_source_segment(source, round_function)
assert training.count(".998**(r-1)") == 1
training = compile(training.replace(".998**(r-1)", ".998**((r-1)*200/1000)"),
                   "train_pair_fedavg", "exec")
start = source.index("a=p.parse_args()")
end = source.index("seed_all(a.seed);m=make_model()")
setup = compile(source[:start] + "a=ARGS\n" + source[start + len("a=p.parse_args()\n"):end],
                "train_pair_setup", "exec")
if args.check_only:
    import hashlib

    partition = reference / "partition.json"
    design = json.loads((reference / "design.json").read_text())
    assert hashlib.sha256(partition.read_bytes()).hexdigest() == design["partition_sha256"]
    assert design["requests"][request]["clients"] == list(range(args.request_size))
    assert len(json.loads(partition.read_text())) == 100
    print(f"Validated seed {seed} launcher, fixed partition, and request size {request}")
    raise SystemExit(0)

out.mkdir(parents=True, exist_ok=True)
ns = dict(__file__=str(root / "train_pair.py"),
          ARGS=SimpleNamespace(dataset="cifar100", regime="dirichlet01", seed=seed,
                               rounds=2000, out=out, smoke=False, pilot=False,
                               extract=False, role="Original"))
exec(setup, ns)
exec(training, ns)

helpers = ast.parse(Path(__file__).with_name("partial_helpers.py").read_text())
nodes = [node for node in helpers.body if isinstance(node, ast.FunctionDef)
         and node.name in ("sampled", "write_json")]
assert len(nodes) == 2
exec(compile(ast.Module(body=nodes, type_ignores=[]), "reference_helpers", "exec"))
partition = reference / "partition.json"
design = json.loads((reference / "design.json").read_text())
assert ns["digest"](partition) == design["partition_sha256"]
(out / "partition.json").write_bytes(partition.read_bytes())
parts = {int(k): np.asarray(v, dtype=np.int64) for k, v in json.loads(partition.read_text()).items()}
ns["parts"] = parts
targets = design["requests"][request]["clients"]
assert targets == list(range(args.request_size))
target_design = out / "design.json"
requests = json.loads(target_design.read_text())["requests"] if target_design.exists() else {}
requests[request] = {"clients": targets}
write_json(out / "design.json", dict(seed=seed, partition_seed=2026,
           partition_sha256=design["partition_sha256"], training_rounds=2000,
           requests=requests))
ns["seed_all"](seed)
model = ns["make_model"]()
initial = ns["cpu_state"](model)

for role, clients in (("Original", sorted(parts)), (f"Retrain{request}", sorted(set(parts)-set(targets)))):
    final = out / f"{role}.pt"
    if final.exists():
        continue
    model.load_state_dict(initial)
    resume = out / f"{role}-resume.pt"
    first = 1
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state"])
        first = state["round"] + 1
    for r in range(first, 2001):
        selected = sampled(clients, r)
        loss = ns["train_round"](model, selected, r)
        if r % 25 == 0 or r == 1:
            write_json(out / "status.json", dict(phase="training", role=role, round=r, loss=loss))
            ns["atomic_save"](dict(round=r, state=ns["cpu_state"](model)), resume)
    ns["atomic_save"](ns["cpu_state"](model), final)
    resume.unlink(missing_ok=True)
write_json(out / "status.json", dict(phase="complete", seed=seed))
