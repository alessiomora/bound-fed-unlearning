"""Loss-score membership audit with disjoint calibration and evaluation sets."""
import json
import numpy as np
import torch
from torch.nn.functional import cross_entropy


def class_weights(labels, frequencies):
    counts = np.bincount(labels, minlength=len(frequencies))
    if np.any((frequencies > 0) & (counts == 0)):
        raise ValueError("Missing class support in MIA split")
    return frequencies[labels] / counts[labels]


def summarize(forget, calibration, unseen, labels, target_fpr=0.05):
    frequencies = np.bincount(labels[0], minlength=max(100, 1 + max(int(y.max()) for y in labels))) / len(forget)
    cw = class_weights(labels[1], frequencies)
    nw = class_weights(labels[2], frequencies)
    grid, index = np.unique(calibration, return_inverse=True)
    cumulative = np.cumsum(np.bincount(index, weights=cw))
    eligible = grid[cumulative <= target_fpr]
    threshold = float(eligible[-1]) if len(eligible) else float(np.nextafter(grid[0], -np.inf))
    # Each positive wins against negatives with higher loss, with half credit for ties.
    order = np.argsort(unseen)
    negative_losses = unseen[order]
    weights = np.r_[0., np.cumsum(nw[order])]
    left = np.searchsorted(negative_losses, forget, side="left")
    right = np.searchsorted(negative_losses, forget, side="right")
    auc = np.mean(weights[-1] - (weights[left] + weights[right]) / 2)
    return dict(roc_auc=float(auc), threshold=threshold, target_fpr=target_fpr,
                calibration_fpr=float(cw[calibration <= threshold].sum()),
                unseen_fpr=float(nw[unseen <= threshold].sum()),
                forget_flag_rate=float(np.mean(forget <= threshold)))


@torch.inference_mode()
def evaluate_mia(model, batches, forget, split, out, round_number):
    model.eval()
    scores, labels = [], []
    groups = (forget, split["anchors"], split["unseen"])
    assert not np.intersect1d(groups[1], groups[2]).size
    for ids, dataset in zip(groups, ("train", "test", "test")):
        losses, targets = [], []
        for x, y in batches(ids, False, None, dataset):
            losses.append(cross_entropy(model(x), y, reduction="none").double().cpu().numpy())
            targets.append(y.cpu().numpy())
        scores.append(np.concatenate(losses))
        labels.append(np.concatenate(targets))
    assert all(np.isfinite(s).all() for s in scores)
    result = summarize(*scores, labels)
    result["n"] = dict(zip(("forget", "calibration", "unseen"), map(len, groups)))
    np.savez_compressed(out / f"mia_losses_{round_number:02d}.npz",
                        **{f"{key}_{kind}": value
                           for key, values in zip(("forget", "calibration", "unseen"),
                                                  zip(groups, labels, scores))
                           for kind, value in zip(("ids", "labels", "losses"), values)})
    path = out / f"mia_{round_number:02d}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(path)
