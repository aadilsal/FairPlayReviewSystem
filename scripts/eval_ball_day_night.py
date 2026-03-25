import argparse
import json
import math
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Ultralytics is required. Install with: pip install ultralytics\n"
        f"Import error: {e}"
    )

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install with: pip install pyyaml\n"
        f"Import error: {e}"
    )


COCO_2017_VAL_IMAGES_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
COCO_2017_VAL_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category id for "sports ball"
COCO_SPORTS_BALL_CAT_ID = 37


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _download(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    _ensure_dir(out_path.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or "0")
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {out_path.name}") as pbar:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))


def _unzip(zip_path: Path, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    marker = out_dir / ".unzipped"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")


def _load_coco_instances(instances_json: Path) -> Dict:
    with open(instances_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _xywh_to_box(x: float, y: float, w: float, h: float) -> Box:
    return Box(x1=x, y1=y, x2=x + w, y2=y + h)


def _iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _mean_luma_bgr(img_bgr: np.ndarray) -> float:
    # ITU-R BT.601 luma approximation (Y' from BGR -> RGB)
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return float(y.mean())


@dataclass
class PresenceConfusion:
    # Confusion matrix for "ball present" at image-level:
    #   GT absent/present vs Pred absent/present
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    def add(self, gt_present: bool, pred_present: bool) -> None:
        if gt_present and pred_present:
            self.tp += 1
        elif gt_present and (not pred_present):
            self.fn += 1
        elif (not gt_present) and pred_present:
            self.fp += 1
        else:
            self.tn += 1

    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return float(2 * p * r / (p + r)) if (p + r) else 0.0

    def accuracy(self) -> float:
        total = self.tn + self.fp + self.fn + self.tp
        return float((self.tp + self.tn) / total) if total else 0.0

    def as_matrix(self) -> List[List[int]]:
        # Rows: GT [absent, present]
        # Cols: Pred [absent, present]
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def to_dict(self) -> Dict:
        return {
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "accuracy": self.accuracy(),
        }


def _print_matrix(title: str, c: PresenceConfusion) -> None:
    m = c.as_matrix()
    print(f"\n{title}")
    print("Confusion matrix (image-level ball presence)")
    print("           Pred: Absent   Pred: Present")
    print(f"GT: Absent      {m[0][0]:>6}        {m[0][1]:>6}")
    print(f"GT: Present     {m[1][0]:>6}        {m[1][1]:>6}")
    print(
        "Metrics: "
        f"precision={c.precision():.3f}  recall={c.recall():.3f}  f1={c.f1():.3f}  accuracy={c.accuracy():.3f}"
    )

def _save_confusion_png(title: str, c: PresenceConfusion, out_path: Path) -> None:
    """
    Save a nice confusion-matrix visualization (PNG) without requiring seaborn.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[WARN] Matplotlib not available, cannot save confusion plot: {e}")
        return

    m = np.array(c.as_matrix(), dtype=np.int64)
    # Normalize for color scale, but keep raw counts as text
    denom = m.sum() if m.sum() > 0 else 1
    norm = m.astype(np.float32) / float(denom)

    fig = plt.figure(figsize=(6.5, 5.0), dpi=200)
    ax = plt.gca()
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=max(0.001, float(norm.max())))

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_xticks([0, 1], labels=["Absent", "Present"])
    ax.set_yticks([0, 1], labels=["Absent", "Present"])

    # annotate raw counts
    for (i, j), val in np.ndenumerate(m):
        ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=11, color="black")

    # add metrics line
    metrics = (
        f"precision={c.precision():.3f}  recall={c.recall():.3f}  "
        f"f1={c.f1():.3f}  accuracy={c.accuracy():.3f}"
    )
    ax.text(0.5, -0.18, metrics, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of samples")
    fig.tight_layout()

    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _pick_best_pred_box(result, conf_min: float) -> Optional[Box]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_conf = -1.0
    for b in boxes:
        conf = float(b.conf[0].item()) if hasattr(b.conf, "__len__") else float(b.conf.item())
        if conf < conf_min:
            continue
        xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
        pred_box = Box(x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]))
        if conf > best_conf:
            best_conf = conf
            best = pred_box
    return best


def _model_ball_class_id(model: YOLO) -> Optional[int]:
    """
    Try to infer which class id corresponds to 'ball' in the *model* metadata.
    If not found, return None (meaning: don't filter by class at prediction time).
    """
    names = getattr(model, "names", None)
    if names is None:
        return None
    if isinstance(names, dict):
        iterator = names.items()
    else:
        iterator = enumerate(names)
    for cid, name in iterator:
        if "ball" in str(name).lower():
            return int(cid)
    return None


def _pred_boxes(result, conf_min: float, class_filter: Optional[int]) -> List[Box]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    out: List[Box] = []
    for b in boxes:
        conf = float(b.conf[0].item()) if hasattr(b.conf, "__len__") else float(b.conf.item())
        if conf < conf_min:
            continue
        if class_filter is not None:
            cls = int(b.cls[0].item()) if hasattr(b.cls, "__len__") else int(b.cls.item())
            if cls != class_filter:
                continue
        xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
        out.append(Box(x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3])))
    return out


def _any_match(preds: List[Box], gts: List[Box], iou_thr: float) -> bool:
    if not preds or not gts:
        return False
    for p in preds:
        for g in gts:
            if _iou(p, g) >= iou_thr:
                return True
    return False


def _load_yolo_data_yaml(dataset_root: Path) -> Dict:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at: {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_split_images_dir(dataset_root: Path, data: Dict, split: str) -> Path:
    if split not in ("train", "val", "valid", "test"):
        raise ValueError("split must be one of: train, val, valid, test")

    key = "val" if split in ("val", "valid") else split
    rel = data.get(key, None)
    if rel is None:
        raise KeyError(f"`{key}` not found in data.yaml")

    # Roboflow exports often use paths like ../train/images. In some repos, that can resolve to
    # "<repo_root>/train/images" (outside the dataset folder) which may exist but be empty.
    # Prefer an images directory that actually contains image files.

    candidates: List[Path] = []

    # 1) Resolve relative to dataset_root (location of data.yaml).
    candidates.append((dataset_root / str(rel)).resolve())

    # 2) Resolve relative to dataset_root.parent (older exports sometimes assume that).
    candidates.append((dataset_root.parent / str(rel)).resolve())

    # 3) Always try the canonical Roboflow layout inside dataset_root.
    #    split "valid"/"val" -> folder "valid"
    split_folder = "valid" if split in ("val", "valid") else split
    candidates.append((dataset_root / split_folder / "images").resolve())

    # De-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)

    # Prefer an existing directory that actually has at least 1 image.
    for c in uniq:
        if c.exists() and c.is_dir():
            imgs = _iter_image_paths(c)
            if len(imgs) > 0:
                return c

    # Otherwise return the first existing directory (even if empty) for clearer errors upstream.
    for c in uniq:
        if c.exists() and c.is_dir():
            return c

    return uniq[0]


def _iter_image_paths(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: List[Path] = []
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if Path(fn).suffix.lower() in exts:
                paths.append(Path(root) / fn)
    paths.sort()
    return paths


def _image_to_label_path(img_path: Path) -> Path:
    # Typical YOLOv8 layout: <split>/images/<name>.<ext> and <split>/labels/<name>.txt
    parts = list(img_path.parts)
    try:
        idx = [p.lower() for p in parts].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return img_path.with_suffix(".txt")


def _yolo_norm_to_box(line: str, img_w: int, img_h: int) -> Optional[Tuple[int, Box]]:
    # YOLO label: cls xc yc w h (all normalized)
    s = line.strip()
    if not s:
        return None
    toks = s.split()
    if len(toks) < 5:
        return None
    cls_id = int(float(toks[0]))
    xc = float(toks[1])
    yc = float(toks[2])
    w = float(toks[3])
    h = float(toks[4])
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return cls_id, Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def _find_ball_class_id(data: Dict) -> Optional[int]:
    names = data.get("names", None)
    if names is None:
        return None
    if isinstance(names, dict):
        items = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda x: x[0])
        id_name = items
    else:
        id_name = list(enumerate([str(n) for n in names]))

    for cid, name in id_name:
        if "ball" in name.lower():
            return int(cid)
    return None


def _load_gt_boxes_for_image(img_path: Path, ball_class_id: Optional[int]) -> List[Box]:
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    img_h, img_w = img.shape[:2]

    label_path = _image_to_label_path(img_path)
    if not label_path.exists():
        return []

    boxes: List[Box] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _yolo_norm_to_box(line, img_w=img_w, img_h=img_h)
            if parsed is None:
                continue
            cls_id, box = parsed
            if ball_class_id is None or cls_id == ball_class_id:
                boxes.append(box)
    return boxes


def _build_yolo_day_night_samples(
    dataset_root: Path,
    split: str,
    night_luma_thresh: float,
    max_images: int,
    seed: int,
) -> Tuple[List[Tuple[Path, List[Box], bool]], List[Tuple[Path, List[Box], bool]]]:
    random.seed(seed)
    data = _load_yolo_data_yaml(dataset_root)
    images_dir = _resolve_split_images_dir(dataset_root, data, split=split)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    ball_class_id = _find_ball_class_id(data)
    image_paths = _iter_image_paths(images_dir)
    if max_images > 0 and len(image_paths) > max_images:
        image_paths = random.sample(image_paths, k=max_images)

    samples: List[Tuple[Path, List[Box], bool]] = []
    for img_path in tqdm(image_paths, desc=f"Preparing {split} samples", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        luma = _mean_luma_bgr(img)
        is_night = luma < night_luma_thresh
        gt_boxes = _load_gt_boxes_for_image(img_path, ball_class_id=ball_class_id)
        samples.append((img_path, gt_boxes, is_night))

    day = [s for s in samples if not s[2]]
    night = [s for s in samples if s[2]]
    return day, night


def _evaluate_images(
    model: YOLO,
    images: List[Tuple[Path, List[Box], bool]],
    iou_thr: float,
    conf_min: float,
) -> PresenceConfusion:
    """
    Each item: (image_path, gt_boxes, is_night)
    We evaluate image-level "ball present" using:
      - GT present if gt_boxes non-empty
      - Pred present if model predicts any box above conf_min
      - A TP requires pred_present and IoU(best_pred, any_gt) >= iou_thr when GT present
        (if GT absent, any pred_present counts as FP)
    """
    c = PresenceConfusion()
    ball_cls = _model_ball_class_id(model)

    for img_path, gt_boxes, _ in tqdm(images, desc="Evaluating", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_present = len(gt_boxes) > 0

        # Filter to ball class if the model exposes a 'ball' class in its metadata.
        # This prevents stump/other classes from being counted as "ball present".
        predict_kwargs = {"verbose": False}
        if ball_cls is not None:
            predict_kwargs["classes"] = [ball_cls]

        results = model.predict(img, **predict_kwargs)
        result = results[0] if results else None
        preds = _pred_boxes(result, conf_min=conf_min, class_filter=ball_cls) if result is not None else []
        pred_present = len(preds) > 0

        if not gt_present:
            c.add(gt_present=False, pred_present=pred_present)
            continue

        if not pred_present:
            c.add(gt_present=True, pred_present=False)
            continue

        # For presence-at-image-level, if ANY predicted ball box overlaps ANY GT ball box,
        # count as TP. This is fairer than only checking the single best confidence box.
        if _any_match(preds, gt_boxes, iou_thr=iou_thr):
            c.add(gt_present=True, pred_present=True)
        else:
            c.add(gt_present=True, pred_present=False)

    return c


def _build_coco_subset(
    coco_root: Path,
    max_pos: int,
    max_neg: int,
    night_luma_thresh: float,
    seed: int,
) -> Tuple[List[Tuple[Path, List[Box], bool]], List[Tuple[Path, List[Box], bool]]]:
    """
    Returns (day_samples, night_samples) where each sample is:
      (image_path, gt_boxes_for_sports_ball, is_night)
    We sample:
      - positives: images containing COCO sports ball annotations
      - negatives: images without sports ball annotations
    Night/day is inferred by mean luma threshold on the *actual image*.
    """
    random.seed(seed)

    ann_json = coco_root / "annotations" / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017" / "val2017"

    coco = _load_coco_instances(ann_json)

    images_by_id: Dict[int, Dict] = {int(im["id"]): im for im in coco["images"]}
    anns = coco["annotations"]

    ball_anns: Dict[int, List[Box]] = {}
    for a in anns:
        if int(a.get("category_id", -1)) != COCO_SPORTS_BALL_CAT_ID:
            continue
        img_id = int(a["image_id"])
        x, y, w, h = a["bbox"]
        ball_anns.setdefault(img_id, []).append(_xywh_to_box(float(x), float(y), float(w), float(h)))

    pos_ids = list(ball_anns.keys())
    random.shuffle(pos_ids)
    pos_ids = pos_ids[: max_pos if max_pos > 0 else len(pos_ids)]

    all_ids = list(images_by_id.keys())
    neg_ids = [i for i in all_ids if i not in ball_anns]
    random.shuffle(neg_ids)
    neg_ids = neg_ids[: max_neg if max_neg > 0 else len(neg_ids)]

    def to_sample(img_id: int, gt_boxes: List[Box]) -> Optional[Tuple[Path, List[Box], bool]]:
        file_name = images_by_id[img_id]["file_name"]
        p = img_dir / file_name
        img = cv2.imread(str(p))
        if img is None:
            return None
        luma = _mean_luma_bgr(img)
        is_night = luma < night_luma_thresh
        return (p, gt_boxes, is_night)

    all_samples: List[Tuple[Path, List[Box], bool]] = []
    for img_id in tqdm(pos_ids, desc="Preparing positives", unit="img"):
        s = to_sample(img_id, ball_anns.get(img_id, []))
        if s is not None:
            all_samples.append(s)

    for img_id in tqdm(neg_ids, desc="Preparing negatives", unit="img"):
        s = to_sample(img_id, [])
        if s is not None:
            all_samples.append(s)

    day = [s for s in all_samples if not s[2]]
    night = [s for s in all_samples if s[2]]
    return day, night


def _prepare_coco(coco_root: Path) -> None:
    _ensure_dir(coco_root)
    z_img = coco_root / "val2017.zip"
    z_ann = coco_root / "annotations_trainval2017.zip"

    _download(COCO_2017_VAL_IMAGES_ZIP, z_img)
    _download(COCO_2017_VAL_ANN_ZIP, z_ann)

    _unzip(z_img, coco_root / "val2017")
    _unzip(z_ann, coco_root / "annotations")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate your ball model and compare DAY vs NIGHT performance.\n"
            "DAY/NIGHT split is inferred from image brightness (mean luma threshold).\n"
            "By default this script uses a local YOLOv8 dataset (Roboflow export)."
        )
    )
    ap.add_argument("--weights", default="weights/ball-yolov8s.pt", help="Path to your ball YOLO weights.")
    ap.add_argument(
        "--dataset-root",
        default="Cricket Dataset.v1i.yolov8",
        help="Path to your YOLOv8 dataset root (must contain data.yaml).",
    )
    ap.add_argument("--split", default="valid", choices=["train", "valid", "val", "test"], help="Which split to test.")
    ap.add_argument("--max-images", type=int, default=0, help="Optional cap on number of images (0 = all).")
    ap.add_argument("--night-luma-thresh", type=float, default=70.0, help="Mean luma below this => night.")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a correct detection.")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for counting a prediction.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducibility.")
    ap.add_argument("--out-dir", default="artifacts/ball_day_night_eval", help="Where to save plots/metrics.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    weights_path = (repo_root / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights)
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    print("Loading model.")
    model = YOLO(str(weights_path))

    print(f"Reading dataset from: {dataset_root}")
    print("Building day/night splits (brightness-based).")
    day, night = _build_yolo_day_night_samples(
        dataset_root=dataset_root,
        split=args.split,
        night_luma_thresh=args.night_luma_thresh,
        max_images=args.max_images,
        seed=args.seed,
    )

    if len(day) == 0 and len(night) == 0:
        print("No samples collected (images missing or unreadable).")
        return 2

    print(f"Samples: day={len(day)}  night={len(night)}  (night threshold luma<{args.night_luma_thresh})")

    summary: Dict = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "weights": str(weights_path),
        "night_luma_thresh": float(args.night_luma_thresh),
        "iou": float(args.iou),
        "conf": float(args.conf),
        "samples": {"day": len(day), "night": len(night)},
        "results": {},
    }

    if day:
        day_c = _evaluate_images(model=model, images=day, iou_thr=args.iou, conf_min=args.conf)
        _print_matrix("DAY results", day_c)
        _save_confusion_png("DAY (ball presence)", day_c, out_dir / "confusion_day.png")
        summary["results"]["day"] = day_c.to_dict()
    else:
        print("\nDAY results\nNo day samples under current threshold.")
        summary["results"]["day"] = None

    if night:
        night_c = _evaluate_images(model=model, images=night, iou_thr=args.iou, conf_min=args.conf)
        _print_matrix("NIGHT results", night_c)
        _save_confusion_png("NIGHT (ball presence)", night_c, out_dir / "confusion_night.png")
        summary["results"]["night"] = night_c.to_dict()
    else:
        print("\nNIGHT results\nNo night samples under current threshold.")
        summary["results"]["night"] = None

    _ensure_dir(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'confusion_day.png'}")
    print(f"Saved: {out_dir / 'confusion_night.png'}")
    print(f"Saved: {out_dir / 'metrics.json'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

