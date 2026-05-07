"""
VisDrone-ReID Leave-One-Out Evaluation Script
==============================================
Evaluates a ReID model on the VisDrone-ReID test split using a
leave-one-out (LOO) protocol

Protocol
--------
For each image i in the test set:
  - Query   = image i
  - Gallery = every other image in the dataset (image i excluded)
  - Correct matches = gallery images whose pid == query pid

This gives one query per image, so an identity with N images
contributes N queries. mAP and CMC are averaged across all queries.

Metrics reported
----------------
  Accuracy  : Rank-1, Rank-5, Rank-10, mAP
  Efficiency: feature extraction time, per-image latency,
              query time, gallery memory, model params, FLOPs

Usage
-----
    python evaluate_visdrone_reid_loo.py `
        --dataset_root C:\path\to\VisDroneReID `
        --weights      C:\path\to\model.pth `
        --output_dir   C:\path\to\output
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    import torchreid
except ImportError:
    sys.exit("torchreid not found. Activate your torchreid conda environment.")

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("[Warning] thop not installed — FLOPs will not be reported.")


# =============================================================================
# 1.  Data loading
# =============================================================================

def parse_pid(fname: str) -> int:
    """Parse identity id from filename: {pid:06d}_{class}_{idx:04d}.jpg"""
    stem  = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse filename: {fname}")
    return int(parts[0])


def load_test_images(dataset_root: str):
    """
    Walk dataset_root/test/ and return a flat ordered list of
    (abs_path, pid) for all identities with >= 2 images.

    Returns
    -------
    all_data     : list of (abs_path, pid)
    n_identities : int
    """
    test_dir = os.path.join(dataset_root, "test")
    if not os.path.isdir(test_dir):
        sys.exit(f"Test directory not found: {test_dir}")

    identity_images = defaultdict(list)

    for pid_folder in sorted(os.listdir(test_dir)):
        pid_path = os.path.join(test_dir, pid_folder)
        if not os.path.isdir(pid_path):
            continue
        for fname in sorted(os.listdir(pid_path)):
            if not fname.lower().endswith(".jpg"):
                continue
            try:
                pid = parse_pid(fname)
            except ValueError as e:
                print(f"[Warning] Skipping {fname}: {e}")
                continue
            identity_images[pid].append(os.path.join(pid_path, fname))

    valid_pids = {p for p, imgs in identity_images.items() if len(imgs) >= 2}
    removed    = len(identity_images) - len(valid_pids)
    if removed:
        print(f"[Data] Removed {removed} identities with < 2 images")

    all_data = []
    for pid in sorted(valid_pids):
        for path in identity_images[pid]:
            all_data.append((path, pid))

    n_identities = len(valid_pids)
    n_images     = len(all_data)

    print(f"[Data] {n_images} images, {n_identities} identities, "
          f"avg {n_images/n_identities:.1f} images/identity")

    return all_data, n_identities


# =============================================================================
# 2.  PyTorch Dataset
# =============================================================================

class ReIDImageDataset(Dataset):
    def __init__(self, data, transform):
        self.data      = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, pid = self.data[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), pid


# =============================================================================
# 3.  Model loading
# =============================================================================

def load_osnet_ibn(weights_path: str, device: torch.device) -> nn.Module:
    """
    Build OSNet-IBN from torchreid and load checkpoint weights.

    Handles two checkpoint formats:
      Format 1 — torchreid training checkpoint:
                 {"state_dict": {...}, "epoch": N, ...}
      Format 2 — raw state dict (pretrained weights):
                 {"layer.weight": tensor, ...}

    Reads num_classes from classifier.weight shape to avoid size mismatch.
    Strips the 'module.' prefix added by DataParallel if present.
    """
    print(f"[Model] Loading: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Detect checkpoint (ckpt) format
    if isinstance(ckpt, dict) and "state_dict" in ckpt:  # for .pth.tar ckpt
        state_dict = ckpt["state_dict"]
        epoch      = ckpt.get("epoch", "?")
    else:
        # Raw state dict — no wrapper dict
        state_dict = ckpt   # for .pth ckpt
        epoch      = "pretrained"

    # Strip 'module.' prefix from DataParallel checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        cleaned[new_key] = v

    # Read num_classes from classifier shape
    clf_key = "classifier.weight"
    if clf_key not in cleaned:
        raise KeyError(
            f"Cannot find '{clf_key}' in checkpoint. "
            f"Available keys (first 10): {list(cleaned.keys())[:10]}"
        )
    num_classes = cleaned[clf_key].shape[0]
    print(f"[Model] Building osnet_ibn_x1_0 with num_classes={num_classes}")

    model = torchreid.models.build_model(
        name="osnet_ibn_x1_0", num_classes=num_classes, pretrained=False
    )
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    model.to(device)
    print(f"[Model] Loaded — Epoch: {epoch}")
    return model


# =============================================================================
# 4.  Feature extraction
# =============================================================================

@torch.no_grad()
def extract_features(model, all_data, transform, device,
                     batch_size=64, num_workers=0):
    """
    Extract L2-normalised embeddings for all images in one pass.

    Returns
    -------
    features : np.ndarray [N, D]
    pids     : np.ndarray [N]
    elapsed  : float  (seconds)
    """
    dataset = ReIDImageDataset(all_data, transform)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers,
                         pin_memory=(device.type == "cuda"))

    all_feats = []
    all_pids  = []

    t0 = time.perf_counter()
    for imgs, pids in loader:
        imgs  = imgs.to(device)
        feats = model(imgs)
        feats = nn.functional.normalize(feats, p=2, dim=1)
        all_feats.append(feats.cpu().numpy())
        all_pids.extend(pids.tolist())
    elapsed = time.perf_counter() - t0

    features = np.concatenate(all_feats, axis=0)
    pids     = np.array(all_pids, dtype=np.int64)
    return features, pids, elapsed


# =============================================================================
# 5.  Evaluation
# =============================================================================

def compute_AP(match_vector: np.ndarray) -> float:
    """
    Average Precision for one query.
    match_vector: binary array [N-1] in ranked order (1=correct, 0=wrong).
    """
    num_rel = match_vector.sum()
    if num_rel == 0:
        return 0.0
    cumsum  = match_vector.cumsum()
    prec_at = np.array([c / (i + 1.) for i, c in enumerate(cumsum)])
    return float((prec_at * match_vector).sum() / num_rel)


def evaluate_loo(features: np.ndarray,
                 pids:     np.ndarray,
                 max_rank: int = 10):
    """
    Leave-one-out (LOO) evaluation with explicit gallery rebuild.

    For each query i:
      - Gallery = all images except image i  (size N-1)
      - Distances computed from query i to all gallery images
      - Gallery ranked by ascending cosine distance
      - Correct matches = gallery images with same pid as query

    Returns
    -------
    cmc     : np.ndarray [max_rank]
    mAP     : float
    elapsed : float  (seconds, evaluation only)
    """
    N       = len(pids)
    all_cmc = []
    all_AP  = []

    t0 = time.perf_counter()

    for i in range(N):
        q_feat = features[i]      # [D]
        q_pid  = pids[i]

        # Gallery: every image except image i
        gallery_idx = np.concatenate([np.arange(0, i),
                                      np.arange(i + 1, N)])
        g_feats = features[gallery_idx]   # [N-1, D]
        g_pids  = pids[gallery_idx]       # [N-1]

        # Cosine distances to gallery
        dists = 1.0 - g_feats @ q_feat    # [N-1]

        # Rank gallery by distance (ascending)
        order   = np.argsort(dists)
        matches = (g_pids[order] == q_pid).astype(np.int32)

        if not np.any(matches):
            continue

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # AP
        all_AP.append(compute_AP(matches))

    elapsed = time.perf_counter() - t0

    all_cmc = np.array(all_cmc, dtype=np.float32).mean(axis=0)
    mAP     = float(np.mean(all_AP))

    return all_cmc, mAP, elapsed


# =============================================================================
# 6.  Efficiency metrics
# =============================================================================

def compute_efficiency_metrics(model, device, features):
    metrics = {}

    metrics["params_M"] = round(
        sum(p.numel() for p in model.parameters()) / 1e6, 3
    )

    if THOP_AVAILABLE:
        try:
            dummy = torch.zeros(1, 3, 256, 128).to(device)
            flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            metrics["flops_G"] = round(flops / 1e9, 3)
        except Exception as e:
            metrics["flops_G"] = f"error: {e}"
    else:
        metrics["flops_G"] = "thop not installed"

    metrics["feature_mem_MB"] = round(features.nbytes / (1024 ** 2), 3)

    return metrics


# =============================================================================
# 7.  Report
# =============================================================================

def print_report(cmc, mAP, extract_time, eval_time,
                 n_images, n_identities, eff):

    sep = "=" * 60

    print(f"\n{sep}")
    print("  VisDrone-ReID  Leave-One-Out Results")
    print(f"{sep}")

    print(f"\n  Images         : {n_images}")
    print(f"  Identities     : {n_identities}")
    print(f"  Avg images/id  : {n_images/n_identities:.1f}")

    print("\n── Accuracy ──────────────────────────────────────────")
    print(f"  Rank-1  : {cmc[0]*100:.2f}%")
    print(f"  Rank-5  : {cmc[4]*100:.2f}%")
    print(f"  Rank-10 : {cmc[9]*100:.2f}%")
    print(f"  mAP     : {mAP*100:.2f}%")

    print("\n── Efficiency ────────────────────────────────────────")
    print(f"  Model params          : {eff['params_M']} M")
    print(f"  Model FLOPs (1 img)   : {eff['flops_G']} GFLOPs")
    print(f"  Feature matrix mem    : {eff['feature_mem_MB']} MB")
    print(f"\n  Feature extraction")
    print(f"    Total               : {extract_time:.3f} s")
    print(f"    Per image           : {extract_time/n_images*1000:.2f} ms")
    print(f"\n  LOO evaluation")
    print(f"    Total               : {eval_time:.3f} s")
    print(f"    Per query           : {eval_time/n_images*1000:.2f} ms")

    print(f"\n{sep}\n")


# =============================================================================
# 8.  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OSNet-IBN on VisDrone-ReID (LOO protocol)"
    )
    parser.add_argument("--dataset_root", type=str, default='./VisDroneReID_data',
                        help="Root of VisDroneReID (contains test/ folder)")
    parser.add_argument("--weights",      type=str, default='./pretrained/osnet_ibn_x1_0_imagenet.pth',
                        help="Path to model checkpoint or pretrained weights")
    parser.add_argument("--output_dir",   type=str,
                        default="./eval_results_loo",
                        help="Directory to save JSON results")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=0,
                        help="0 is safest on Windows")
    parser.add_argument("--device",       type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available — falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[Device] {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    # ── Step 1: load data ─────────────────────────────────────────────────
    print("\n[1/4] Loading test images...")
    all_data, n_identities = load_test_images(args.dataset_root)
    n_images = len(all_data)
    if n_images == 0:
        sys.exit("No valid images found. Check --dataset_root.")

    # ── Step 2: model ─────────────────────────────────────────────────────
    print("\n[2/4] Loading model...")
    model = load_osnet_ibn(args.weights, device)

    # ── Step 3: extract features once ────────────────────────────────────
    print(f"\n[3/4] Extracting features ({n_images} images)...")
    features, pids, extract_time = extract_features(
        model, all_data, transform, device,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"  Done in {extract_time:.2f} s")

    eff = compute_efficiency_metrics(model, device, features)

    # ── Step 4: evaluate ─────────────────────────────────────────────────
    print(f"\n[4/4] Evaluating ({n_images} queries, LOO)...")
    cmc, mAP, eval_time = evaluate_loo(features, pids, max_rank=10)
    print(f"  Done in {eval_time:.2f} s")

    # ── Report ────────────────────────────────────────────────────────────
    print_report(cmc, mAP, extract_time, eval_time,
                 n_images, n_identities, eff)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "dataset":  "VisDrone-ReID",
        "protocol": "Leave-One-Out (explicit gallery rebuild)",
        "model":    "osnet_ibn_x1_0",
        "accuracy": {
            "rank1":  round(float(cmc[0]) * 100, 2),
            "rank5":  round(float(cmc[4]) * 100, 2),
            "rank10": round(float(cmc[9]) * 100, 2),
            "mAP":    round(mAP * 100, 2),
        },
        "efficiency": {
            **eff,
            "n_images":             n_images,
            "n_identities":         n_identities,
            "avg_images_per_id":    round(n_images / n_identities, 1),
            "extract_time_s":       round(extract_time, 4),
            "per_image_extract_ms": round(extract_time / n_images * 1000, 3),
            "eval_time_s":          round(eval_time, 4),
            "per_query_ms":         round(eval_time / n_images * 1000, 3),
        },
        "config": vars(args)
    }

    out_path = os.path.join(args.output_dir, "visdrone_reid_loo_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {out_path}")


# Execute from the interpreter
if __name__ == "__main__":
    main()