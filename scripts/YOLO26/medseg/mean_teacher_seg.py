"""Image-only Mean-Teacher data, pseudo-target, and EMA helpers for polyp adaptation."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywh

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(s: str):
    if s.lower() == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if s.isdigit():
        return torch.device(f"cuda:{s}")

    return torch.device(s)


def make_divisible(x: int, divisor: int = 32):
    return int(math.ceil(float(x) / divisor) * divisor)


def list_images(root: Path):
    images = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: (int(p.stem) if p.stem.isdigit() else p.stem),
    )

    if not images:
        raise RuntimeError(f"No images found: {root}")

    return images


class TargetMTDataset(Dataset):

    def __init__(
        self,
        images: list[Path],
        imgsz: int,
    ):
        self.images = images
        self.imgsz = imgsz

    def __len__(self):
        return len(self.images)

    @staticmethod
    def strong_photo(im: np.ndarray):
        """
        Strong PHOTOMETRIC augmentation only.

        Geometry remains identical to weak image.
        Teacher and Student share geometry, so pseudo masks remain aligned.
        """

        out = im.copy()

        if random.random() < 0.8:
            hsv = cv2.cvtColor(
                out,
                cv2.COLOR_RGB2HSV,
            ).astype(np.float32)

            hsv[..., 0] = np.clip(
                hsv[..., 0] + random.uniform(-0.10, 0.10) * 180,
                0,
                179,
            )

            hsv[..., 1] = np.clip(
                hsv[..., 1] + random.uniform(-0.20, 0.20) * 255,
                0,
                255,
            )

            hsv[..., 2] = np.clip(
                hsv[..., 2] + random.uniform(-0.20, 0.20) * 255,
                0,
                255,
            )

            out = cv2.cvtColor(
                hsv.astype(np.uint8),
                cv2.COLOR_HSV2RGB,
            )

        if random.random() < 0.5:
            alpha = random.uniform(
                0.85,
                1.15,
            )

            beta = random.uniform(
                -15,
                15,
            )

            out = np.clip(
                alpha * out.astype(np.float32) + beta,
                0,
                255,
            ).astype(np.uint8)

        if random.random() < 0.25:
            kernel = random.choice([3, 5])

            out = cv2.GaussianBlur(
                out,
                (kernel, kernel),
                0,
            )

        return np.ascontiguousarray(out)

    def __getitem__(self, idx):
        path = self.images[idx]

        im = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if im is None:
            raise RuntimeError(path)

        im = cv2.cvtColor(
            im,
            cv2.COLOR_BGR2RGB,
        )

        h, w = im.shape[:2]

        scale = self.imgsz / float(max(h, w))

        nh = int(round(h * scale))
        nw = int(round(w * scale))

        im = cv2.resize(
            im,
            (nw, nh),
            interpolation=cv2.INTER_LINEAR,
        )

        # Shared geometry for Teacher and Student.
        if random.random() < 0.5:
            im = np.ascontiguousarray(np.fliplr(im))

        weak = im.copy()
        strong = self.strong_photo(im)

        weak = torch.from_numpy(np.ascontiguousarray(weak)).permute(2, 0, 1).float() / 255.0

        strong = torch.from_numpy(np.ascontiguousarray(strong)).permute(2, 0, 1).float() / 255.0

        return weak, strong, str(path)


def collate(batch):
    weak, strong, paths = zip(*batch)

    max_h = make_divisible(
        max(x.shape[1] for x in weak),
        32,
    )

    max_w = make_divisible(
        max(x.shape[2] for x in weak),
        32,
    )

    def pad(images):
        result = []

        for x in images:
            c, h, w = x.shape

            canvas = torch.full(
                (c, max_h, max_w),
                114.0 / 255.0,
                dtype=x.dtype,
            )

            canvas[:, :h, :w] = x

            result.append(canvas)

        return torch.stack(
            result,
            dim=0,
        )

    return (
        pad(weak),
        pad(strong),
        list(paths),
    )


def ensure_seg_loss_args(
    model: nn.Module,
    epochs: int,
):
    existing = getattr(
        model,
        "args",
        None,
    )

    if existing is None:
        existing = argparse.Namespace()

    elif isinstance(existing, dict):
        existing = argparse.Namespace(**existing)

    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "epochs": int(epochs),
        # Simpler representation:
        # one binary mask per pseudo instance.
        "overlap_mask": False,
    }

    for key, value in defaults.items():
        setattr(
            existing,
            key,
            value,
        )

    model.args = existing


def setup_teacher_student(
    checkpoint: str,
    device: torch.device,
    epochs: int,
):
    teacher_wrapper = YOLO(checkpoint)

    teacher = teacher_wrapper.model.to(device).float()

    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    student_wrapper = YOLO(checkpoint)

    student = student_wrapper.model.to(device).float()

    ensure_seg_loss_args(
        student,
        epochs,
    )

    student.criterion = student.init_criterion()

    for p in student.parameters():
        p.requires_grad_(True)

    head = student.model[-1]

    if type(head).__name__ != "Segment26":
        raise RuntimeError(f"Expected Segment26, got {type(head).__name__}")

    if not getattr(
        student,
        "end2end",
        False,
    ):
        raise RuntimeError("Student is not end-to-end")

    return (
        teacher,
        student,
        student_wrapper,
        student.criterion,
    )


def build_pseudo_batch(
    labels: list[torch.Tensor],
    masks: list[torch.Tensor],
    input_shape,
):
    """
    Build pseudo targets for native YOLO26 segmentation loss.

    Source-free:
      - boxes come from Teacher pseudo predictions
      - instance masks come from Teacher mask coefficients + proto
      - semantic masks are constructed from those pseudo instances
      - NO target GT is used
    """

    device = labels[0].device

    h = int(input_shape[2])
    w = int(input_shape[3])

    norm = torch.tensor(
        [w, h, w, h],
        device=device,
        dtype=torch.float32,
    )

    all_batch_idx = []
    all_cls = []
    all_boxes = []
    all_masks = []

    for img_idx, (
        rows,
        instance_masks,
    ) in enumerate(zip(labels, masks)):
        if rows.numel() == 0:
            continue

        if rows.shape[0] != instance_masks.shape[0]:
            raise RuntimeError("Pseudo box/mask count mismatch")

        n = rows.shape[0]

        all_batch_idx.append(
            torch.full(
                (n,),
                img_idx,
                device=device,
                dtype=torch.long,
            )
        )

        all_cls.append(rows[:, 5].long())

        xywh = xyxy2xywh(rows[:, :4])

        all_boxes.append(xywh / norm)

        all_masks.append(instance_masks.float())

    if not all_boxes:
        return None

    batch_idx = torch.cat(
        all_batch_idx,
        dim=0,
    )

    cls = torch.cat(
        all_cls,
        dim=0,
    )

    bboxes = torch.cat(
        all_boxes,
        dim=0,
    )

    instance_masks = torch.cat(
        all_masks,
        dim=0,
    )

    # ------------------------------------------------------------
    # Proto26 semantic branch
    # ------------------------------------------------------------
    #
    # Current medical task:
    #     nc = 1
    #     class 0 = polyp
    #
    # v8SegmentationLoss converts sem_masks with:
    #
    #     F.one_hot(sem_masks, num_classes=nc)
    #
    # and, when overlap_mask=False, suppresses pixels outside
    # the instance pseudo masks.
    #
    # Therefore for the single-class polyp task, every semantic
    # foreground pixel has class id 0. A zero-valued class-index
    # map is correct; the pseudo instance masks determine which
    # pixels remain active.
    #
    # This uses NO target GT.
    # ------------------------------------------------------------

    if torch.any(cls != 0):
        raise RuntimeError("Pseudo-target construction requires the single-class " "polyp setting with class id 0.")

    mask_h = int(instance_masks.shape[-2])

    mask_w = int(instance_masks.shape[-1])

    sem_masks = torch.zeros(
        (
            len(labels),
            mask_h,
            mask_w,
        ),
        device=device,
        dtype=torch.long,
    )

    batch = {
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        # overlap_mask=False:
        # [total_instances, Hm, Wm]
        "masks": instance_masks,
        # Proto26 semantic branch:
        # [batch_size, Hm, Wm]
        "sem_masks": sem_masks,
    }

    return batch


@torch.no_grad()
def update_teacher_ema(
    teacher: nn.Module,
    student: nn.Module,
    momentum: float,
):
    for tp, sp in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        tp.data.mul_(momentum).add_(
            sp.data,
            alpha=1.0 - momentum,
        )


class TargetCalibrationDataset(Dataset):
    """
    Target image-only dataset.

    No labels and no masks are accessible here.
    """

    def __init__(
        self,
        images: list[Path],
        imgsz: int,
    ):
        self.images = images
        self.imgsz = imgsz

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]

        im = cv2.imread(str(path))

        if im is None:
            raise RuntimeError(path)

        im = cv2.cvtColor(
            im,
            cv2.COLOR_BGR2RGB,
        )

        h, w = im.shape[:2]

        scale = self.imgsz / float(max(h, w))

        nh = int(round(h * scale))
        nw = int(round(w * scale))

        im = cv2.resize(
            im,
            (nw, nh),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0

        return tensor, str(path)


def collate_calibration(batch):
    images, paths = zip(*batch)

    max_h = make_divisible(
        max(x.shape[1] for x in images),
        32,
    )

    max_w = make_divisible(
        max(x.shape[2] for x in images),
        32,
    )

    padded = []

    for x in images:
        c, h, w = x.shape

        canvas = torch.full(
            (c, max_h, max_w),
            114.0 / 255.0,
            dtype=x.dtype,
        )

        canvas[:, :h, :w] = x

        padded.append(canvas)

    return torch.stack(padded), list(paths)
