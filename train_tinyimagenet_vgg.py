"""Paired clean-client removal benchmark on Tiny ImageNet-200."""
import argparse
import fcntl
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        inputs = 3
        for channels, count in [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]:
            for _ in range(count):
                layers += [nn.Conv2d(inputs, channels, 3, padding=1),
                           nn.GroupNorm(8, channels), nn.ReLU(inplace=True)]
                inputs = channels
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, 200)
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def encode(self, images):
        return self.features(images).mean((2, 3))

    def forward(self, images):
        return self.fc(self.encode(images))


def index_dataset(data):
    classes = sorted((data / "wnids.txt").read_text().split())
    assert len(classes) == len(set(classes)) == 200
    names = {name: i for i, name in enumerate(classes)}
    train = [(path, names[name]) for name in classes
             for path in sorted((data / "train" / name / "images").glob("*.JPEG"))]
    rows = [line.split() for line in (data / "val/val_annotations.txt").read_text().splitlines()]
    heldout = [(data / "val/images" / row[0], names[row[1]]) for row in sorted(rows)]
    assert len(train) == 100_000 and len(heldout) == 10_000
    assert all(path.is_file() for path, _ in train + heldout)
    return train, heldout


def make_splits(labels, heldout, alpha=.1):
    rng = np.random.default_rng(2026)
    clients = [[] for _ in range(10)]
    splits = {name: [] for name in ("validation", "anchors", "unseen")}
    for label in range(200):
        ids = rng.permutation(np.flatnonzero(labels == label))
        counts = rng.multinomial(len(ids), rng.dirichlet(np.full(10, alpha)))
        for client, chunk in zip(clients, np.split(ids, np.cumsum(counts)[:-1])):
            client.extend(chunk.tolist())
        ids = rng.permutation(np.flatnonzero(heldout == label))
        for name, chunk in zip(splits, np.split(ids, [10, 30])):
            splits[name].extend(chunk.tolist())
    assert all(len(ids) > 0 for ids in clients)
    return clients, splits


def read_images(records):
    result = np.empty((len(records), 64, 64, 3), dtype=np.uint8)
    for index, (path, _) in enumerate(records):
        with Image.open(path) as image:
            result[index] = np.asarray(image.convert("RGB"))
    return torch.from_numpy(result).cuda()


def atomic_save(state, path):
    required = sum(value.numel() * value.element_size() for value in state["model"].values())
    if shutil.disk_usage(path.parent).free < 2_000_000_000 + required + 1_000_000:
        raise RuntimeError("Insufficient space above the 2 GB disk reserve")
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--scratch-checkpoints", type=Path)
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    train, heldout = index_dataset(args.data)
    labels = np.asarray([label for _, label in train])
    val_labels = np.asarray([label for _, label in heldout])
    clients, splits = make_splits(labels, val_labels, args.alpha)
    if args.alpha != .1:
        _, splits = make_splits(labels, val_labels)
    scratch = args.scratch_checkpoints or args.out
    print(json.dumps({"client_sizes": list(map(len, clients)),
                      "heldout_sizes": {k: len(v) for k, v in splits.items()}}), flush=True)
    if args.index_only:
        return

    lock = (Path(__file__).resolve().parent / "gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    args.out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    config = dict(dataset="Tiny ImageNet-200", model="VGG-16 convolutional blocks + GN8 + GAP + linear",
                  seed=args.seed, rounds=args.rounds, clients=10, alpha=args.alpha, target=0,
                  lr_peak=.03, lr_floor=.001, warmup_rounds=5, batch_size=64,
                  local_epochs=1, momentum=.9, weight_decay=.0005,
                  dataset_index_sha256=hashlib.sha256("\n".join(
                      f"{path.relative_to(args.data)} {label}" for path, label in train + heldout
                  ).encode()).hexdigest())
    manifest = args.out / "protocol.json"
    text = json.dumps(dict(config=config, partition=clients, heldout=splits), indent=2)
    if manifest.exists():
        assert manifest.read_text() == text, "Existing protocol differs"
    else:
        manifest.write_text(text)

    train_x, val_x = read_images(train), read_images(heldout)
    train_y, val_y = torch.as_tensor(labels, device="cuda"), torch.as_tensor(val_labels, device="cuda")
    clients = [torch.tensor(ids, device="cuda") for ids in clients]
    val_ids = torch.tensor(splits["validation"], device="cuda")
    pixels = torch.arange(64, device="cuda")

    def images(buffer, ids, generator=None):
        x = buffer[ids].permute(0, 3, 1, 2).float().div_(255)
        if generator is not None:
            padded = F.pad(x, (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
            offset = torch.randint(9, (len(x), 2), generator=generator, device="cuda")
            x = padded[torch.arange(len(x), device="cuda")[:, None, None],
                       pixels[None, :, None] + offset[:, 0, None, None],
                       pixels[None, None, :] + offset[:, 1, None, None]].permute(0, 3, 1, 2)
            flip = torch.rand(len(x), generator=generator, device="cuda") < .5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return (x - .5) / .25

    torch.manual_seed(args.seed)
    model = VGG16().cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for role, retained in (("Original", list(range(10))), ("Retrain", list(range(1, 10)))):
        final = args.out / f"{role}.pt"
        if final.exists():
            continue
        resume = scratch / f"{role}-resume.pt"
        milestone = scratch / f"{role}-round100.pt"
        model.load_state_dict(initial)
        first = 1
        if resume.exists():
            saved = torch.load(resume, map_location="cuda", weights_only=True)
            model.load_state_dict(saved["model"])
            first = saved["round"] + 1
            del saved
        total = sum(len(clients[c]) for c in retained)
        for r in range(first, args.rounds + 1):
            started = time.monotonic()
            lr = .03 * r / 5 if r <= 5 else .001 + .029 * (1 + math.cos(math.pi * (r - 5) / max(1, args.rounds - 5))) / 2
            global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            average = {k: torch.zeros_like(v) for k, v in global_state.items()}
            for c in retained:
                model.load_state_dict(global_state)
                model.train()
                opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=.0005)
                generator = torch.Generator(device="cuda").manual_seed(args.seed + 10000 * r + c)
                ids = clients[c][torch.randperm(len(clients[c]), generator=generator, device="cuda")]
                for batch in ids.split(64):
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(images(train_x, batch, generator)), train_y[batch])
                    assert torch.isfinite(loss), (role, r, c)
                    loss.backward()
                    opt.step()
                for name, value in model.state_dict().items():
                    average[name].add_(value, alpha=len(ids) / total)
            model.load_state_dict(average)
            model.eval()
            with torch.inference_mode():
                correct = sum((model(images(val_x, batch)).argmax(1) == val_y[batch]).sum().item()
                              for batch in val_ids.split(128))
            record = dict(role=role, round=r, lr=lr, validation_accuracy=correct / len(val_ids),
                          seconds=time.monotonic() - started)
            print(json.dumps(record), flush=True)
            with (args.out / "progress.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            if r == 100 and args.rounds > 100:
                atomic_save(dict(round=r, model=model.state_dict()), milestone)
            if r % 5 == 0 or r == args.rounds:
                atomic_save(dict(round=r, model=model.state_dict()), resume)
        atomic_save(dict(round=args.rounds, model=model.state_dict()), final)
        resume.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
