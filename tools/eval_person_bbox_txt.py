#!/usr/bin/env python3
"""
Evaluate person detection (class 0) using YOLO txt labels.

Ground-truth format per line:
  <class> <xc> <yc> <w> <h>

Prediction format per line:
  <class> <conf> <xc> <yc> <w> <h>
or (without confidence):
  <class> <xc> <yc> <w> <h>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Box:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = xc - (w / 2.0)
    y1 = yc - (h / 2.0)
    x2 = xc + (w / 2.0)
    y2 = yc + (h / 2.0)
    return x1, y1, x2, y2


def iou(a: Box, b: Box) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def parse_gt(path: Path) -> List[Box]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.strip().split()
        if len(p) != 5:
            continue
        cls = int(float(p[0]))
        xc, yc, w, h = map(float, p[1:])
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h)
        out.append(Box(cls=cls, conf=1.0, x1=x1, y1=y1, x2=x2, y2=y2))
    return out


def parse_pred(path: Path) -> List[Box]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.strip().split()
        if len(p) == 6:
            cls = int(float(p[0]))
            conf = float(p[1])
            xc, yc, w, h = map(float, p[2:])
        elif len(p) == 5:
            cls = int(float(p[0]))
            conf = 1.0
            xc, yc, w, h = map(float, p[1:])
        else:
            continue
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h)
        out.append(Box(cls=cls, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate person detection from YOLO txt files.")
    parser.add_argument("--gt-dir", required=True, help="Directory with ground-truth .txt files")
    parser.add_argument("--pred-dir", required=True, help="Directory with prediction .txt files")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    names = sorted(p.stem for p in gt_dir.glob("*.txt"))

    tp = 0
    fp = 0
    fn = 0
    gt_person_total = 0
    pred_person_total = 0

    for name in names:
        gt = [b for b in parse_gt(gt_dir / f"{name}.txt") if b.cls == 0]
        pred = [b for b in parse_pred(pred_dir / f"{name}.txt") if b.cls == 0 and b.conf >= args.conf]
        pred.sort(key=lambda b: b.conf, reverse=True)

        gt_person_total += len(gt)
        pred_person_total += len(pred)

        matched = [False] * len(gt)
        for pb in pred:
            best_i = -1
            best_iou = 0.0
            for i, gb in enumerate(gt):
                if matched[i]:
                    continue
                ov = iou(pb, gb)
                if ov > best_iou:
                    best_iou = ov
                    best_i = i
            if best_i >= 0 and best_iou >= args.iou:
                matched[best_i] = True
                tp += 1
            else:
                fp += 1

        fn += sum(1 for m in matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("=== Person Detection Metrics ===")
    print(f"Files evaluated         : {len(names)}")
    print(f"GT person boxes         : {gt_person_total}")
    print(f"Pred person boxes       : {pred_person_total}")
    print(f"TP / FP / FN            : {tp} / {fp} / {fn}")
    print(f"IoU threshold           : {args.iou:.2f}")
    print(f"Confidence threshold    : {args.conf:.2f}")
    print(f"precision@{args.iou:.2f}       : {precision:.4f}")
    print(f"recall@{args.iou:.2f}          : {recall:.4f}")
    print(f"f1@{args.iou:.2f}              : {f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

