"""Fetch public data for the published CIFAR-100 and Stanford Cars splits."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from torchvision.datasets import CIFAR100

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cifar100", "stanfordcars"))
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    if args.dataset == "cifar100":
        for training in (True, False):
            CIFAR100(args.data, train=training, download=True)
        return

    manifest = json.loads((ROOT / "prepared/stanfordcars/mirror.json").read_text())
    destination = args.data / "stanford-cars-parquet"
    destination.mkdir(exist_ok=True)
    for item in manifest["files"]:
        filename = item["path"]
        cached = Path(hf_hub_download(
            manifest["dataset"], f"data/{filename}", repo_type="dataset",
            revision=manifest["revision"],
        ))
        with cached.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or cached.stat().st_size != item["bytes"]:
            raise ValueError(f"Stanford Cars integrity check failed for {filename}")
        link = destination / filename
        if link.is_symlink():
            if link.resolve() != cached.resolve():
                raise FileExistsError(link)
        else:
            link.symlink_to(cached.resolve())


if __name__ == "__main__":
    main()
