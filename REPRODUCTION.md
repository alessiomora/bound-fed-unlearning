# Reproduction check

The released CIFAR runner was executed on one NVIDIA A10G using PyTorch
2.5.1+cu121 and torchvision 0.20.1+cu121. It loaded the archived IID
seed-2026 Original checkpoint, fitted BOUND for ten local epochs
(full encoder, feature MSE, beta 2, learning rate 0.00025), and
completed one retain-only FedAvg recovery round. This checks the
unlearning and recovery path from an existing reference checkpoint.

| Round | Metric (%) | Archived | Replay | Replay minus archived (pp) |
| --- | --- | ---: | ---: | ---: |
| 0 | FA | 25.82 | 25.90 | +0.08 |
| 0 | RA | 75.52 | 75.34 | -0.18 |
| 0 | TA | 50.85 | 50.82 | -0.03 |
| 0 | Probe FA | 41.84 | 41.94 | +0.10 |
| 1 | FA | 47.00 | 46.48 | -0.52 |
| 1 | RA | 84.61 | 84.64 | +0.03 |
| 1 | TA | 58.88 | 59.13 | +0.25 |
| 1 | Probe FA | 49.78 | 49.62 | -0.16 |

Round-0 loss MIA AUROC was 40.55 in the archive and 40.56 in the
replay on the 0–100 scale. The run completed and wrote both round
metrics and both MIA outputs. The comparison check used a tolerance
of 0.50 percentage points and flagged round-1 FA, which differed by
0.52 points. Full reference retraining and round-20 endpoint replay
were outside this verification run.

To repeat the check, use the CIFAR BOUND command from the README with
`--rounds 1` and a fresh output directory. With access to the archived
checkpoint, its SHA-256 is
`e63b891486117cb5c653f74fa1bb7daef17c776a70876adf62004b3117d59b2f`.

The pinned Stanford Cars mirror matched all 8,144 training and 8,041
test image hashes in archive order. The 100-client fixed partition and
five-request manifest passed the reference runner's `--check-only`
validation. The Cars and Tiny ImageNet training and recovery runs
have not been rerun as part of this release check.
