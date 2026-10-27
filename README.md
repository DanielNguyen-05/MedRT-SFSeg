# MedRT-SFSeg

**Real-time polyp segmentation with source-free domain adaptation.**

**Accepted at MMM 2027 (International Conference on Multimedia Modeling).**

**Paper:** *Directional Dual-Head Supervision for Real-Time Source-Free Polyp Segmentation*

**Authors:** Hai-Dang Nguyen, Nhat-Khoi Le, Huu-Thuan Huynh, Minh-Triet Tran, and Trung-Nghia Le

**Links:** [Project page](https://danielnguyen-05.github.io/MedRT-SFSeg/) · [Citation](#citation)

MedRT-SFSeg is a research project built on YOLO26-S-Seg and a local Ultralytics codebase. It studies how a segmentation model trained on one endoscopic dataset can adapt to another using only the source-trained checkpoint and unlabeled target images. Mask-DHF, DURR, and Seg-MARD provide training supervision while preserving the model's native O2O inference path.

The main application is **polyp segmentation** across **Kvasir-SEG** and **CVC-ClinicDB**, with experiments in both directions:

- **K2C:** Kvasir-SEG → CVC-ClinicDB.
- **C2K:** CVC-ClinicDB → Kvasir-SEG.

## Background

Medical images can differ in appearance across datasets, imaging devices, and acquisition conditions. This **domain shift** can reduce the performance of a model when it is applied to a new dataset.

In **source-free domain adaptation**, a model first learns from labeled source data. During adaptation, it uses its trained checkpoint and unlabeled target images, without accessing the original source data. **Pseudo-labels** are predictions used as training targets in place of manual annotations. Target ground-truth masks are reserved for evaluation and analysis.

## Main approach

The workflow combines supervised source training and adaptation to unlabeled target images:

```text
Source training → AdaBN → Mean Teacher + Mask-DHF + DURR + Seg-MARD
```

| Component | Purpose |
| --- | --- |
| **YOLO26-S-Seg** | Base segmentation model with one-to-one (O2O) and one-to-many (O2M) prediction branches. |
| **AdaBN** | Updates batch normalization statistics using target images. |
| **Mean Teacher** | Uses an exponential moving average (EMA) of student weights to provide teacher predictions for adaptation. |
| **Mask-DHF** | Combines predictions from both branches and filters pseudo-masks using confidence and mask reliability. |
| **DURR** | Dual-head Uncertainty Reliability Routing: uses agreement and disagreement between branches to guide boundary refinement, recover reliable missed regions, and suppress unsupported predictions. |
| **Seg-MARD* | Uses pseudo-masks to sample foreground and background features for variance and covariance regularization. |

DURR and Seg-MARD are used during training. The adapted model retains the YOLO26 O2O inference path.

## Repository structure

```text
MedRT-SFSeg/
├── scripts/YOLO26/
│   └── medseg/          # Polyp data preparation, adaptation, and evaluation
├── ultralytics/         # Local YOLO implementation
└── pyproject.toml       # Package metadata and dependencies
```

Datasets are typically stored under `dataset/`, while checkpoints, metrics, and visualizations are written under `runs/seg/`. Prepared segmentation datasets contain images, YOLO polygon labels, ground-truth masks, and a `dataset_seg.yaml` file.

## Basic commands

Run these commands from the repository root. The examples below illustrate the Kvasir-SEG → CVC-ClinicDB workflow and assume that the required datasets and starting weights are available. They use CUDA device `0`; adjust paths, batch sizes, and device settings for your environment.

**Set up the local environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export PYTHONPATH="$PWD"
```

**Prepare the polyp datasets:**

```bash
python scripts/YOLO26/medseg/prepare_kvasir.py \
  --src dataset/Kvasir-SEG --dst dataset/Kvasir-SEG-YOLO26

python scripts/YOLO26/medseg/prepare_cvc_clinicdb.py \
  --src dataset/CVC-ClinicDB/PNG --dst dataset/CVC-ClinicDB-YOLO26
```

**Train a source model:**

```bash
python scripts/YOLO26/medseg/train_source_seg.py \
  --weights yolo26s-seg.pt \
  --data dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml \
  --epochs 100 --device 0
```

**Adapt batch normalization, then train on unlabeled target images:**

```bash
python scripts/YOLO26/medseg/stage1_adabn_seg.py \
  --weights runs/seg/source/kvasir_yolo26s_seg/weights/best.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/stage1/cvc_adabn --device 0

python scripts/YOLO26/medseg/stage2_dense_sfseg_durr.py \
  --weights runs/seg/stage1/cvc_adabn/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/cvc_durr \
  --epochs 60 --device 0
```

**Evaluate the adapted student:**

```bash
python scripts/YOLO26/medseg/evaluate_durr_final.py \
  --model runs/seg/dense_sfseg/cvc_durr/checkpoints/durr_student_epoch_60.pt \
  --data dataset/CVC-ClinicDB-YOLO26/dataset_seg.yaml \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/benchmarks/cvc_durr --device 0
```

Use `--help` with an entry-point script to view its available options.

## Paper implementation

The scripts in `scripts/YOLO26/medseg/` cover the method and component ablations described in the paper:

| Scripts | Role |
| --- | --- |
| `prepare_kvasir.py`, `prepare_cvc_clinicdb.py`, `prepare_cvc_to_kvasir_reverse.py` | Prepare the datasets for both adaptation directions. |
| `train_source_seg.py`, `stage1_adabn_seg.py` | Train the source model and update target BatchNorm statistics. |
| `mean_teacher_seg.py` | Shared image-only datasets, pseudo-target construction, and EMA updates. |
| `mask_dhf_seg.py`, `durr_seg.py`, `segmard_seg.py` | Mask-DHF, directional routing, and mask-guided feature regularization. |
| `calibrate_mask_reliability.py` | Estimate the target-only reliability threshold from the initial teacher. |
| `stage2_dense_sfseg.py` | Mask-DHF and Mask-DHF + Seg-MARD ablations. |
| `stage2_dense_sfseg_durr.py` | Full MedRT-SFSeg and the Mask-DHF + DURR ablation. |
| `eval_source_seg.py`, `evaluate_durr_final.py` | Evaluate source-only, ablation, and adapted checkpoints. |
| `analyze_seg_results.py`, `benchmark_seg_speed.py`, `qualitative_o2o_o2m_panel.py` | Segmentation metrics, boundary visualizations, and inference speed. |

For each adaptation direction, run `calibrate_mask_reliability.py` with the AdaBN checkpoint and target images. Its JSON output includes `suggested_mask_rel_thr`, the 25th percentile of candidate mask reliability; pass this value as `--mask-rel-thr` when adapting. The example training command above uses the K2C default.

To disable Seg-MARD in either adaptation trainer, set `--mard-lambda0 0 --mard-lambda-max 0`. This provides the Mask-DHF and Mask-DHF + DURR configurations in the paper's ablation study.

## Evaluation and analysis

The evaluation tools report **Dice**, **Intersection over Union (IoU)**, **mask mAP**, precision, and recall. Boundary analysis includes **Average Surface Distance (ASD)** and the **95th-percentile Hausdorff Distance (HD95)**. Speed benchmarks measure latency and frames per second (FPS).

The analysis tools visualize predicted masks, ground-truth contours, and boundary errors. The full adaptation trainer saves student and EMA teacher checkpoints along with experiment metadata. Ground-truth masks are used only for evaluation and optional post-training visualizations.

## Citation

If you use MedRT-SFSeg in your research, please cite our paper:

```bibtex
@inproceedings{NguyenMMM2027,
  title = {Directional Dual-Head Supervision for Real-Time Source-Free Polyp Segmentation},
  author = {Nguyen, Hai-Dang and Le, Nhat-Khoi and Huynh, Huu-Thuan and Tran, Minh-Triet and Le, Trung-Nghia},
  booktitle = {International Conference on Multimedia Modeling (MMM)},
  year = {2027},
}
```

This citation follows the [project page](https://danielnguyen-05.github.io/MedRT-SFSeg/#BibTeX). Full proceedings details will be added when available.
