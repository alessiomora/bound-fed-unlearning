# BOUND for federated client unlearning

This repository contains the experiment code for BOUND, paired Original and retained-only Retrain references, and retain-only recovery. It includes the fixed client partitions and evaluation splits for CIFAR-100 and Stanford Cars. The Tiny ImageNet runner regenerates its partition and split deterministically. Each BOUND intervention starts from Original and uses only the requesting clients' data. Recovery starts after the intervention, with the requesting clients excluded.

## Setup

Use Linux, Python 3.11 or 3.12, and a CUDA-capable GPU with enough memory to hold cached images and the model. The experiments used 24 GB GPUs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python prepare_data.py cifar100
python prepare_data.py stanfordcars
```

The Cars downloader verifies the four public mirror shards at the pinned revision. The model loader verifies the experiment's image order by SHA-256 and downloads the pinned ImageNet-pretrained DeiT-T weights. Set `BOUND_DATA` to reuse a CIFAR-100 or Cars data directory with the same layout. For Tiny ImageNet, extract the 200-class release and point `--data` at the directory containing `wnids.txt`, `train/`, and `val/`.

## CIFAR-100 / ResNet-18, ten clients

Repeat for `regime=iid` and `regime=dirichlet01`, and training seeds 2026, 2027, and 2028. Training uses 200 rounds by default. The requesting client is client 0.

```bash
regime=iid
seed=2026
ref="runs/cifar100_${regime}_s${seed}"
python train_pair.py --dataset cifar100 --regime "$regime" --seed "$seed" --out "$ref"
for role in Original Retrain BOUND; do
  source_role=Original
  if [ "$role" = Retrain ]; then source_role=Retrain; fi
  python unlearning/run_bound_ablation20.py --role "$role" --regime "$regime" \
    --seed "$seed" --source "$ref/$source_role.pt" --name "$role" \
    --out "$ref/eval-$role" --scope encoder --objective feature \
    --beta 2 --lr 0.00025 --epochs 10 --rounds 20 --loss-mia
done
```

To reproduce the non-IID sensitivity sweep, run the same BOUND command with `--beta 0.25`, `0.5`, `1`, `2`, `3`, and `4` in separate output directories. The runner also exposes final-block training (`--scope block`) and direct margin matching (`--objective margin`).

## Stanford Cars / DeiT-T, ten clients

Use `lr=0.00006` for IID and `lr=0.00005` for Dirichlet-0.1. Reference training takes 100 rounds. The main-table IID row uses seeds 2026 and 2027, and the provisional non-IID BOUND row uses seed 2026.

```bash
regime=iid
seed=2026
lr=0.00006
ref="runs/stanfordcars_${regime}_s${seed}"
python train_pair.py --dataset stanfordcars --regime "$regime" --seed "$seed" --out "$ref"
for role in Original Retrain BOUND; do
  source_role=Original
  if [ "$role" = Retrain ]; then source_role=Retrain; fi
  python unlearning/run_cars.py --role "$role" --regime "$regime" \
    --seed "$seed" --source "$ref/$source_role.pt" --out "$ref/eval-$role" \
    --scope full_encoder --beta 2 --lr "$lr" --epochs 10 --rounds 10
done
```

## Tiny ImageNet / VGG-16, ten clients

The reported tuning result uses seed 2026. The network has VGG-16 convolutional blocks, group normalization, global average pooling, and a linear head.

```bash
data=/path/to/tiny-imagenet-200
ref=runs/tinyimagenet_vgg16_dirichlet03_s2026
python train_tinyimagenet_vgg.py --data "$data" --out "$ref" --rounds 200 --alpha 0.3 --seed 2026
for role in Original Retrain; do
  python run_tinyimagenet_unlearning.py --reference "$ref" --data "$data" \
    --out "$ref/eval-$role" --seed 2026 --job "{\"role\":\"$role\"}"
done
python run_tinyimagenet_unlearning.py --reference "$ref" --data "$data" \
  --out "$ref/eval-BOUND" --seed 2026 \
  --job '{"role":"BOUND","scope":"encoder","beta":2,"lr":0.00009,"epochs":10}'
```

## CIFAR-100 / ResNet-18, 100 clients

The fixed Dirichlet-0.1 partition uses 10 participants per round. Train each reference for 2,000 rounds. The requesting set is client 0 for one withdrawal, and clients 0–4 for five simultaneous withdrawals. BOUND computes a local update per requester from the same Original model and averages updates by the number of eligible forget examples.

```bash
# One withdrawal, 100 recovery rounds. Repeat with --seed 2027 for the two-seed comparison.
seed=2026
python unlearning/train_request5_reference.py --seed "$seed" --request-size 1
for role in Original Retrain BOUND; do
  python unlearning/run_bound100clients50.py --request-size 1 --role "$role" \
    --seed "$seed" --name "request1-${role}-${seed}" --scope encoder \
    --beta 3 --lr 0.0003 --epochs 10 --recovery-rounds 100 --loss-mia
done

# Five withdrawals, 50 recovery rounds.
python unlearning/train_request5_reference.py --seed 2026 --request-size 5
for role in Original Retrain BOUND; do
  python unlearning/run_bound100clients50.py --request-size 5 --role "$role" \
    --seed 2026 --name "request5-${role}-2026" --scope stage \
    --beta 5 --lr 0.001 --epochs 10 --recovery-rounds 50
done
```

## Outputs and verification

FA, RA, and TA are native accuracy on the forget, retain, and unseen test-evaluation sets. Probe FA is forget accuracy of a fresh linear classifier fitted on frozen features using a retained-only fitting subset. Per-round metrics use fractions. Loss MIA is reported as AUROC, with disjoint held-out calibration and evaluation sets. CIFAR and Tiny ImageNet write per-round metric JSON files. Cars writes evaluation entries to `progress.jsonl`.

Data, model weights, and trained checkpoints are acquired separately or generated locally. Outputs go to the specified directory or under `runs/`. See [Reproduction check](REPRODUCTION.md) for the checkpoint replay and its measured differences.
