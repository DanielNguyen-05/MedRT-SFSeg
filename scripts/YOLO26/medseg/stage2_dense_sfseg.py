#!/usr/bin/env python3
"""Mask-DHF and Mask-DHF + Seg-MARD ablations (paper Table 2).

Set --mard-lambda0 0 --mard-lambda-max 0 for Mask-DHF alone.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from mean_teacher_seg import (
    TargetMTDataset,
    build_pseudo_batch,
    collate,
    list_images,
    resolve_device,
    seed_everything,
    setup_teacher_student,
    update_teacher_ema,
)
from mask_dhf_seg import generate_mask_dhf_pseudo_masks
from segmard_seg import add_segmard_args, compute_segmard_loss


class SegmentInputFeatureHook:
    """
    Capture P3/P4/P5 features entering Segment26.
    """

    def __init__(self, model: nn.Module):
        self.latest = None
        self.head = model.model[-1]
        if not (hasattr(self.head, "one2one") and hasattr(self.head, "one2many")):
            raise RuntimeError("Final head is not dual-head")
        self.handle = self.head.register_forward_pre_hook(self._hook)

    @staticmethod
    def _valid_feature_list(x):
        return (
            isinstance(x, (list, tuple))
            and len(x) >= 3
            and all((isinstance(t, torch.Tensor) and t.ndim == 4 for t in x[:3]))
        )

    def _hook(self, _module, inputs):
        if len(inputs) == 1 and self._valid_feature_list(inputs[0]):
            self.latest = list(inputs[0][:3])
        elif self._valid_feature_list(inputs):
            self.latest = list(inputs[:3])
        else:
            self.latest = None

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def mard_weight(args, global_step, steps_per_epoch, avg_conf):
    warmup_steps = max(1, int(args.mard_warmup_epochs * steps_per_epoch))
    ramp = min(1.0, float(global_step) / float(warmup_steps))
    gate = (avg_conf - args.mard_gate_threshold) / max(1.0 - args.mard_gate_threshold, 1e-06)
    gate = float(np.clip(gate, 0.0, 1.0))
    weight = args.mard_lambda0 * ramp * gate
    return min(weight, args.mard_lambda_max)


def average_confidence(labels):
    values = [x[:, 4] for x in labels if x.numel()]
    if not values:
        return 0.0
    return float(torch.cat(values).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--target-images", required=True)
    ap.add_argument("--out-dir", default="runs/seg/dense_sfseg/cvc_dense")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all batches")
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--tau-o2o", type=float, default=0.5)
    ap.add_argument("--tau-o2m", type=float, default=0.5)
    ap.add_argument("--tau-no", type=float, default=0.2)
    ap.add_argument("--tau-dup", type=float, default=0.7)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--stability-low", type=float, default=0.4)
    ap.add_argument("--stability-high", type=float, default=0.6)
    ap.add_argument("--mask-rel-thr", type=float, default=0.744898)
    ap.add_argument("--min-mask-pixels", type=int, default=16)
    ap.add_argument("--mard-lambda0", type=float, default=0.05)
    ap.add_argument("--mard-lambda-max", type=float, default=0.2)
    ap.add_argument("--mard-gamma", type=float, default=1.0)
    ap.add_argument("--mard-alpha", type=float, default=1.0)
    ap.add_argument("--mard-beta", type=float, default=0.1)
    ap.add_argument("--mard-warmup-epochs", type=float, default=5.0)
    ap.add_argument("--mard-gate-threshold", type=float, default=0.5)
    ap.add_argument("--mard-topk-boxes", type=int, default=15)
    ap.add_argument("--mard-fg-points", type=int, default=8)
    ap.add_argument("--mard-bg-points", type=int, default=128)
    ap.add_argument("--mard-eta", type=float, default=12.0)
    ap.add_argument("--mard-box-conf", type=float, default=0.5)
    add_segmard_args(ap)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--print-freq", type=int, default=10)
    ap.add_argument("--save-interval", type=int, default=10)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    images = list_images(Path(args.target_images).resolve())
    dataset = TargetMTDataset(images, args.imgsz)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate,
    )
    teacher, student, student_wrapper, criterion = setup_teacher_student(args.weights, device, epochs=args.epochs)
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.937, weight_decay=0.0005, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    out_dir = Path(args.out_dir).resolve()
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hook = SegmentInputFeatureHook(student)
    print("=" * 72)
    print("DENSE MEDRT-SFSEG STAGE-2")
    print("=" * 72)
    print("target images :", len(images))
    print("labels used   : NO")
    print("GT masks used : NO")
    print("epochs        :", args.epochs)
    print("batch         :", args.batch)
    print()
    print("DHF mode      :", "mask")
    print("Mask-DHF rel  :", args.mask_rel_thr)
    print("stability     :", args.stability_low, args.stability_high)
    print()
    print("MARD lambda0  :", args.mard_lambda0)
    print("MARD warmup   :", args.mard_warmup_epochs)
    print("MARD mode     :", "mask")
    print("FG definition : inside Teacher pseudo segmentation mask")
    print("FG erosion    :", args.segmard_erode_kernel)
    print("mask dilation :", args.segmard_dilate_kernel)
    print("hard BG ratio :", args.segmard_hard_bg_ratio)
    print("reliability wt:", "ON" if args.segmard_reliability_weighting else "OFF")
    print("var/cov obj.   : " + ("reliability-weighted" if args.segmard_reliability_weighting else "unchanged"))
    global_step = 0
    try:
        for epoch in range(args.epochs):
            student.train()
            successful = 0
            skipped = 0
            totals = {
                "loss": 0.0,
                "seg_loss": 0.0,
                "semseg_loss": 0.0,
                "mard": 0.0,
                "lambda": 0.0,
                "pseudo": 0,
                "anchors": 0,
                "box_extras": 0,
                "mask_extras": 0,
                "rejected_rel": 0,
                "fg_tokens": 0.0,
                "hard_bg_tokens": 0.0,
                "easy_bg_tokens": 0.0,
                "core_fallbacks": 0.0,
                "rel_sum": 0.0,
                "rel_count": 0,
                "rel_min": 1.0,
                "rel_max": 0.0,
                "token_weight_sum": 0.0,
            }
            start = time.time()
            for batch_i, (weak, strong, _paths) in enumerate(loader, start=1):
                if args.max_batches > 0 and batch_i > args.max_batches:
                    break
                weak = weak.to(device, non_blocking=True)
                strong = strong.to(device, non_blocking=True)
                labels, geometry_masks, instance_reliabilities, ps = generate_mask_dhf_pseudo_masks(
                    teacher,
                    weak,
                    tau_o2o=args.tau_o2o,
                    tau_o2m=args.tau_o2m,
                    tau_no=args.tau_no,
                    tau_dup=args.tau_dup,
                    mask_threshold=args.mask_thr,
                    stability_low=args.stability_low,
                    stability_high=args.stability_high,
                    reliability_threshold=args.mask_rel_thr,
                    min_mask_pixels=args.min_mask_pixels,
                    return_instance_reliability=True,
                )
                if not len(labels) == len(geometry_masks) == len(instance_reliabilities):
                    raise RuntimeError("Mask-DHF batch output lengths are misaligned")
                for bi_align in range(len(labels)):
                    n_lab = int(labels[bi_align].shape[0])
                    n_geo = int(geometry_masks[bi_align].shape[0])
                    n_rel = int(instance_reliabilities[bi_align].shape[0])
                    if not n_lab == n_geo == n_rel:
                        raise RuntimeError(f"Mask-DHF image {bi_align}: labels={n_lab} geometry={n_geo} rel={n_rel}")
                valid = [i for i, x in enumerate(labels) if x.shape[0] > 0]
                if not valid:
                    skipped += 1
                    global_step += 1
                    continue
                strong_valid = strong[valid]
                labels_valid = [labels[i] for i in valid]
                geometry_masks_valid = [geometry_masks[i] for i in valid]
                reliabilities_valid = [instance_reliabilities[i] for i in valid]
                rel_cat = torch.cat([r.detach().float().flatten() for r in reliabilities_valid if r.numel()], dim=0)
                if rel_cat.numel() == 0:
                    raise RuntimeError("Valid pseudo batch has no aligned reliabilities")
                batch_rel_mean = float(rel_cat.mean().item())
                batch_rel_min = float(rel_cat.min().item())
                batch_rel_max = float(rel_cat.max().item())
                avg_conf = average_confidence(labels_valid)
                hook.latest = None
                student_outputs = student(strong_valid)
                feats = hook.latest
                if feats is None:
                    raise RuntimeError("MARD feature hook captured no features")
                pseudo_batch = build_pseudo_batch(labels_valid, geometry_masks_valid, strong_valid.shape)
                det_vec, loss_items = criterion(student_outputs, pseudo_batch)
                sfseg_loss = det_vec.sum()
                mard_loss, mard_stats = compute_segmard_loss(
                    feats=feats,
                    pseudo_labels=labels_valid,
                    pseudo_masks=geometry_masks_valid,
                    pseudo_reliabilities=reliabilities_valid if args.segmard_reliability_weighting else None,
                    h_pad=int(strong_valid.shape[2]),
                    w_pad=int(strong_valid.shape[3]),
                    args=args,
                )
                lambda_mard = mard_weight(args, global_step, len(loader), avg_conf)
                total_loss = sfseg_loss + lambda_mard * mard_loss
                if not torch.isfinite(total_loss):
                    raise RuntimeError("Non-finite total loss")
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise RuntimeError("Non-finite gradient")
                optimizer.step()
                global_step += 1
                successful += 1
                pseudo_n = sum((x.shape[0] for x in labels_valid))
                totals["loss"] += float(total_loss.detach())
                totals["seg_loss"] += float(loss_items[1].detach())
                totals["semseg_loss"] += float(loss_items[4].detach())
                totals["mard"] += float(mard_loss.detach())
                totals["lambda"] += lambda_mard
                totals["pseudo"] += pseudo_n
                totals["anchors"] += ps["anchors"]
                totals["box_extras"] += ps["box_dhf_extras"]
                totals["mask_extras"] += ps["mask_dhf_extras"]
                totals["rejected_rel"] += ps["rejected_reliability"]
                totals["rel_sum"] += float(rel_cat.sum().item())
                totals["rel_count"] += int(rel_cat.numel())
                totals["rel_min"] = min(totals["rel_min"], batch_rel_min)
                totals["rel_max"] = max(totals["rel_max"], batch_rel_max)
                for key in ("fg_tokens", "hard_bg_tokens", "easy_bg_tokens", "core_fallbacks"):
                    totals[key] += float(mard_stats.get(key, 0.0))
                totals["token_weight_sum"] += float(mard_stats.get("token_weight_mean", 1.0))
                if batch_i == 1 or batch_i % args.print_freq == 0 or batch_i == len(loader):
                    print(
                        f"[E{epoch + 1:02d}] batch={batch_i:03d}/{len(loader):03d} pseudo={pseudo_n} loss={float(total_loss.detach()):.4f} seg={float(loss_items[1]):.4f} semseg={float(loss_items[4]):.4f} mard={float(mard_loss.detach()):.4f} lambda={lambda_mard:.6f} box_extra={ps['box_dhf_extras']} mask_extra={ps['mask_dhf_extras']} "
                        + f"FG/HBG/EBG={int(mard_stats.get('fg_tokens', 0))}/{int(mard_stats.get('hard_bg_tokens', 0))}/{int(mard_stats.get('easy_bg_tokens', 0))} fallback={int(mard_stats.get('core_fallbacks', 0))} rel={batch_rel_mean:.3f}/{batch_rel_min:.3f}/{batch_rel_max:.3f} wmean={float(mard_stats.get('token_weight_mean', 1.0)):.3f} "
                        + f"grad_preclip={float(grad_norm):.2f}",
                        flush=True,
                    )
            if successful == 0:
                raise RuntimeError("Epoch had no valid batches")
            scheduler.step()
            if hasattr(criterion, "update"):
                criterion.update()
            update_teacher_ema(teacher, student, args.ema)
            denom = max(successful, 1)
            print()
            print(
                f"[Epoch {epoch + 1:02d}] DONE time={time.time() - start:.1f}s valid={successful} skipped={skipped} pseudo={totals['pseudo']} anchors={totals['anchors']} box_extra={totals['box_extras']} mask_extra={totals['mask_extras']} reject_rel={totals['rejected_rel']} "
                + f"loss={totals['loss'] / denom:.4f} seg={totals['seg_loss'] / denom:.4f} semseg={totals['semseg_loss'] / denom:.4f} mard={totals['mard'] / denom:.4f} lambda_avg={totals['lambda'] / denom:.6f}"
                + f" FG/HBG/EBG={totals['fg_tokens'] / denom:.1f}/{totals['hard_bg_tokens'] / denom:.1f}/{totals['easy_bg_tokens'] / denom:.1f} rel={totals['rel_sum'] / max(totals['rel_count'], 1):.3f}/{totals['rel_min']:.3f}/{totals['rel_max']:.3f} wmean={totals['token_weight_sum'] / denom:.3f}",
                flush=True,
            )
            save_this_epoch = (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs
            if save_this_epoch:
                hook.close()
                student.eval()
                student_wrapper.model = student
                ckpt = checkpoint_dir / f"dense_sfseg_epoch_{epoch + 1}.pt"
                student_wrapper.save(str(ckpt))
                print("[SAVE]", ckpt, flush=True)
                if epoch + 1 < args.epochs:
                    student.train()
                    hook = SegmentInputFeatureHook(student)
    finally:
        if hook is not None and hook.handle is not None:
            hook.close()
    metadata = {
        "source_free": True,
        "target_labels_used": False,
        "target_gt_masks_used": False,
        "initial_weights": str(Path(args.weights).resolve()),
        "epochs": args.epochs,
        "dhf_mode": "mask",
        "mask_dhf": {
            "tau_o2o": args.tau_o2o,
            "tau_o2m": args.tau_o2m,
            "tau_no": args.tau_no,
            "tau_dup": args.tau_dup,
            "stability_low": args.stability_low,
            "stability_high": args.stability_high,
            "reliability_threshold": args.mask_rel_thr,
            "threshold_source": "Q25 of initial AdaBN Teacher Box-DHF extras; label-free",
        },
        "mard": {
            "lambda0": args.mard_lambda0,
            "lambda_max": args.mard_lambda_max,
            "warmup_epochs": args.mard_warmup_epochs,
            "gamma": args.mard_gamma,
            "alpha": args.mard_alpha,
            "beta": args.mard_beta,
            "fg_points": args.mard_fg_points,
            "bg_points": args.mard_bg_points,
            "eta": args.mard_eta,
            "mode": "mask",
            "foreground_definition": "pseudo segmentation mask",
            "segmard_erode_kernel": args.segmard_erode_kernel,
            "segmard_dilate_kernel": args.segmard_dilate_kernel,
            "segmard_hard_bg_ratio": args.segmard_hard_bg_ratio,
            "segmard_reliability_weighting": args.segmard_reliability_weighting,
            "segmard_reliability_definition": (
                "sqrt(box_confidence * threshold_stability)" if args.segmard_reliability_weighting else "disabled"
            ),
            "segmard_reliability_token_policy": (
                "FG=r_i; HBG=max r_i of covering hard regions; EBG=1"
                if args.segmard_reliability_weighting
                else "uniform"
            ),
        },
        "ema_momentum": args.ema,
        "ema_frequency": "epoch",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print()
    print("[PASS] Dense SFSeg run completed")


if __name__ == "__main__":
    main()
