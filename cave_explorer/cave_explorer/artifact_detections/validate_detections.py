"""
validate_detections.py
--------------------------------------------------------------------------------
Compares localised artifact detections against ground-truth positions.

- Reads two JSON files that live alongside this script:
    1) detections.json
       {
         "artifacts": [
           {"id": 1, "class": "green_crystal", "x": 1.23, "y": 4.56, ...},
           ...
         ]
       }

    2) ground_truths.json
       [
         {"artifact": "artifact_0", "class": "green_crystal",
          "coordinates": {"x": 8.2, "y": 5.2}},
         ...
       ]

- Filters by a chosen set of classes (YOLO label names), configurable below.
- For each selected class, performs a nearest-neighbour matching from detections
  to ground-truths with a distance tolerance (default 0.5 m).
- Prints a detailed per-detection report, including nearest GT and distance.
- Declares PASS only if for every selected class:
    * all detections are matched to a unique ground-truth of the same class,
    * all relevant ground-truths are matched (i.e., no missing),
    * and all matches lie within tolerance.
    
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ==============================
# CONFIGURATION
# ==============================
# Classes to validate (YOLO detector names)
SELECTED_CLASSES: List[str] = ['mushroom', 'green_crystal']

# Match tolerance in meters
TOLERANCE_M: float = 0.5

# Filenames (resolved relative to this script's directory)
DETECTIONS_FILE = "detections.json"
GROUND_TRUTHS_FILE = "ground_truths.json"


# ==============================
# DATA MODELS
# ==============================
@dataclass
class Detection:
    """A single detection as produced by the localisation pipeline."""
    idx: int
    cls: str
    x: float
    y: float

@dataclass
class GroundTruth:
    """A single ground truth artifact entry."""
    name: str
    cls: str
    x: float
    y: float


# ==============================
# UTILS
# ==============================
def euclidean_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return Euclidean distance in the XY plane."""
    return math.hypot(x1 - x2, y1 - y2)


def load_detections(path: Path) -> List[Detection]:
    """
    Load detections from detections.json.

    Expected schema:
        {
          "artifacts": [
            {"id": 1, "class": "green_crystal", "x": 1.23, "y": 4.56, ...},
            ...
          ]
        }
    """
    data = json.loads(path.read_text())
    arts = data.get("artifacts", [])
    dets: List[Detection] = []
    for a in arts:
        try:
            dets.append(Detection(
                idx=int(a.get("id", -1)),
                cls=str(a["class"]),
                x=float(a["x"]),
                y=float(a["y"]),
            ))
        except Exception:
            # Skip malformed entries silently
            continue
    return dets


def load_ground_truths(path: Path) -> List[GroundTruth]:
    """
    Load ground truths from ground_truths.json.

    Expected schema:
        [
          {"artifact": "artifact_0", "class": "green_crystal",
           "coordinates": {"x": 8.2, "y": 5.2}},
          ...
        ]
    """
    arr = json.loads(path.read_text())
    gts: List[GroundTruth] = []
    for item in arr:
        try:
            name = str(item.get("artifact", "artifact"))
            cls = str(item["class"])
            coords = item["coordinates"]
            x = float(coords["x"])
            y = float(coords["y"])
            gts.append(GroundTruth(name=name, cls=cls, x=x, y=y))
        except Exception:
            # Skip malformed entries silently
            continue
    return gts


def group_by_class_detections(dets: List[Detection]) -> Dict[str, List[Detection]]:
    """Group detections by class label."""
    out: Dict[str, List[Detection]] = {}
    for d in dets:
        out.setdefault(d.cls, []).append(d)
    return out


def group_by_class_ground_truths(gts: List[GroundTruth]) -> Dict[str, List[GroundTruth]]:
    """Group ground truths by class label."""
    out: Dict[str, List[GroundTruth]] = {}
    for g in gts:
        out.setdefault(g.cls, []).append(g)
    return out


# ==============================
# MATCHING
# ==============================
def match_detections_to_ground_truths(
    detections: List[Detection],
    truths: List[GroundTruth],
    tolerance: float,
) -> Tuple[List[Tuple[Detection, Optional[GroundTruth], float, bool]], bool]:
    """
    Greedy nearest-neighbour matching from detections to ground truths.

    For each detection:
      - find the nearest, *unmatched* ground truth by Euclidean distance
      - compute distance and pass/fail per tolerance
      - mark GT as used if within tolerance

    Returns:
      - A list of tuples (det, matched_gt_or_None, distance, within_tol_flag)
      - A boolean 'all_matched_and_within_tol' indicating class-level success
        (all detections matched within tol AND all truths matched exactly once).

    Notes:
      - This is a simple greedy approach; for small sets it’s perfectly fine.
      - If you need optimal assignment across all pairs, replace with Hungarian algo.
    """
    results: List[Tuple[Detection, Optional[GroundTruth], float, bool]] = []
    remaining = set(range(len(truths)))  # indices of unmatched GTs

    # 1) Try to match every detection to its nearest GT
    for det in detections:
        best_i = None
        best_d = float("inf")
        for i in remaining:
            gt = truths[i]
            d = euclidean_2d(det.x, det.y, gt.x, gt.y)
            if d < best_d:
                best_d = d
                best_i = i

        if best_i is None:
            # No GTs available
            results.append((det, None, float("inf"), False))
        else:
            gt = truths[best_i]
            within = best_d <= tolerance
            if within:
                remaining.remove(best_i)  # consume GT only if within tolerance
                results.append((det, gt, best_d, True))
            else:
                # Not within tolerance => no match consumed
                results.append((det, gt, best_d, False))

    # 2) Success criteria:
    #    - Every detection got a within-tolerance match (True)
    #    - And all ground-truths were matched exactly once (remaining must be empty)
    all_dets_ok = all(flag for *_ , flag in results)
    all_gts_used = (len(remaining) == 0)

    return results, (all_dets_ok and all_gts_used)


# ==============================
# REPORTING
# ==============================
def print_class_report(
    cls_name: str,
    matches: List[Tuple[Detection, Optional[GroundTruth], float, bool]],
    truths: List[GroundTruth],
    class_pass: bool,
    tolerance: float,
) -> None:
    """Pretty-print a per-class validation report."""
    print("\n" + "=" * 80)
    print(f"Class: {cls_name}")
    print(f"- Tolerance: {tolerance:.3f} m")
    print(f"- Detections: {len(matches)} | Ground truths: {len(truths)}")
    print("-" * 80)
    print(f"{'Det#':>5}  {'Det(x,y)':>22}  {'Nearest GT':>20}  {'GT(x,y)':>22}  {'Dist(m)':>8}  {'OK?':>5}")
    print("-" * 80)

    used_gt_names = set()
    for det, gt, dist, ok in matches:
        det_xy = f"({det.x:.3f}, {det.y:.3f})"
        if gt is None:
            gt_name = "None"
            gt_xy = "(—, —)"
        else:
            gt_name = gt.name
            gt_xy = f"({gt.x:.3f}, {gt.y:.3f})"
            if ok:
                used_gt_names.add(gt.name)

        print(f"{det.idx:>5}  {det_xy:>22}  {gt_name:>20}  {gt_xy:>22}  {dist:>8.3f}  {str(ok):>5}")

    # Summary for this class
    print("-" * 80)
    print(f"Matched GTs (within tol): {len(used_gt_names)}/{len(truths)}")
    print(f"Class PASS: {class_pass}")
    print("=" * 80)


# ==============================
# MAIN
# ==============================
def main() -> None:
    """Load files, filter classes, perform matching, and print a final verdict."""
    script_dir = Path(__file__).resolve().parent
    detections_path = script_dir / DETECTIONS_FILE
    ground_truths_path = script_dir / GROUND_TRUTHS_FILE

    if not detections_path.exists():
        raise FileNotFoundError(f"Missing detections file: {detections_path}")
    if not ground_truths_path.exists():
        raise FileNotFoundError(f"Missing ground truths file: {ground_truths_path}")

    dets_all = load_detections(detections_path)
    gts_all = load_ground_truths(ground_truths_path)

    dets_by_cls = group_by_class_detections(dets_all)
    gts_by_cls = group_by_class_ground_truths(gts_all)

    overall_ok = True
    tested_any = False

    print("\nVALIDATION: Selected classes =", SELECTED_CLASSES)
    print(f"Files:\n - {detections_path}\n - {ground_truths_path}")
    print(f"Tolerance: {TOLERANCE_M:.3f} m")

    for cls in SELECTED_CLASSES:
        # Gather detections and ground-truths for this class
        dets = dets_by_cls.get(cls, [])
        gts = gts_by_cls.get(cls, [])
        tested_any = tested_any or bool(dets or gts)

        # Perform greedy matching
        matches, class_ok = match_detections_to_ground_truths(
            detections=dets,
            truths=gts,
            tolerance=TOLERANCE_M,
        )
        print_class_report(cls, matches, gts, class_ok, TOLERANCE_M)
        overall_ok = overall_ok and class_ok

    if not tested_any:
        print("\nNo detections or ground truths found for the selected classes.")
        overall_ok = False

    # Final verdict
    print("\n" + "#" * 80)
    if overall_ok:
        print("TEST RESULT: PASS ✅  All selected classes matched within tolerance.")
    else:
        print("TEST RESULT: FAIL ❌  Missing/extra detections or out-of-tolerance matches.")
    print("#" * 80 + "\n")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
