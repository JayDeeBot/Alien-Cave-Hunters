#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_label_repair_gui.py

Tkinter GUI to:
- Load images from your YOLO dataset (train/val/test).
- Run detections using a provided 'best.pt' (Ultralytics YOLO).
- Display the per-detection "tensors" [class_id, conf, cx, cy, w, h] (YOLO normalized).
- Allow editing/removing detections *directly with the mouse* on the image.
- Save over the existing label .txt files in the chosen subset directory.

Notes
-----
- Boxes are *stored and saved* in standard YOLO format: "class_id cx cy w h" (no conf).
- Internally, we keep and show a "conf" column to reflect detector confidence if present;
  when loading from .txt (ground truth), conf is set to 1.0 for convenience.
- Coordinates are normalized to [0,1] relative to the image, consistent with YOLO labels.
- The GUI draws rectangles and interactive handles from normalized coords.
- You can switch subset (train/val/test) via a dropdown; paths update accordingly.
- Mouse Editing:
    • Click a box to select it.
    • Drag INSIDE a selected box to move it.
    • Drag a HANDLE (corners/edges) to resize it.
    • Press Delete key (or use Delete button) to remove the selected detection.

Dependencies
------------
pip install ultralytics pillow numpy pyyaml

Author: SpaceRobotics / ChatGPT (Jarred's assistant)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk  # pillow
import numpy as np
import yaml

# ---------------------- CONFIG: update if you change repo layout ----------------------
# IMAGES_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/images/train") # Uncomment for training set 
# LABELS_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/labels/train") # use 80/20 split training/validation

# IMAGES_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/images/val") # Uncomment for validation set
# LABELS_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/labels/val")

IMAGES_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/images/test")  # default to testing set
LABELS_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/labels/test")

CLASSES_YAML = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/tools/classes.yaml")

# Path to your detector weights (Ultralytics best.pt)
MODEL_PATH = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/runs_cave/yolov8n_baseline/weights/best.pt")

# Supported image extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------------------- OPTIONAL: YOLO import (graceful if missing) ------------------
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception as e:
    _HAS_YOLO = False


@dataclass
class Detection:
    """
    Container for a single detection/label entry.

    Fields
    ------
    cls_id : int
        Integer class id per classes.yaml index.
    conf : float
        Confidence (0..1). Saved labels do NOT include conf; conf is shown for UI only.
    cx, cy, w, h : float
        YOLO normalized center-x, center-y, width, height, each in [0,1].
    """
    cls_id: int
    conf: float
    cx: float
    cy: float
    w: float
    h: float

    def to_label_row(self) -> str:
        """Return the line format 'class_id cx cy w h' for YOLO label files."""
        return f"{int(self.cls_id)} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


class LabelStore:
    """
    File I/O helper for reading/writing YOLO label .txt files alongside images.

    Methods
    -------
    load_for_image(img_path: Path) -> List[Detection]
        Loads label .txt for the given image (same stem), if present.
    save_for_image(img_path: Path, dets: List[Detection]) -> None
        Writes/overwrites label .txt with provided detections.
    label_path_for_image(img_path: Path) -> Path
        Compute the label .txt path from an image path using the configured LABELS_DIR.
    """
    def __init__(self, labels_dir: Path):
        self.labels_dir = labels_dir

    def label_path_for_image(self, img_path: Path) -> Path:
        """
        Compute .txt label path for a given image path using the same filename stem.

        Parameters
        ----------
        img_path : Path
            Path to an image file.

        Returns
        -------
        Path
            Path to the corresponding label .txt file in LABELS_DIR.
        """
        return self.labels_dir / (img_path.stem + ".txt")

    def load_for_image(self, img_path: Path) -> List[Detection]:
        """
        Load existing YOLO labels for the given image.

        If no label file exists, returns an empty list.

        Parameters
        ----------
        img_path : Path

        Returns
        -------
        List[Detection]
        """
        lbl_path = self.label_path_for_image(img_path)
        dets: List[Detection] = []
        if not lbl_path.exists():
            return dets

        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        # Skip malformed lines
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:])
                    dets.append(Detection(cls_id=cls_id, conf=1.0, cx=cx, cy=cy, w=w, h=h))
        except Exception as e:
            print(f"[WARN] Failed to read labels for {img_path.name}: {e}", file=sys.stderr)
        return dets

    def save_for_image(self, img_path: Path, dets: List[Detection]) -> None:
        """
        Save (overwrite) YOLO .txt labels for the given image.

        Parameters
        ----------
        img_path : Path
        dets : List[Detection]
        """
        lbl_path = self.label_path_for_image(img_path)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lbl_path, "w", encoding="utf-8") as f:
            for d in dets:
                f.write(d.to_label_row() + "\n")


class ClassMap:
    """
    Loads and provides mapping between class indices and names from classes.yaml.

    YAML format expected:
      names:
        0: class_a
        1: class_b
        ...

    Methods
    -------
    id_to_name(idx: int) -> str
        Returns class name (or f'class_{idx}' if missing).
    """
    def __init__(self, classes_yaml: Path):
        self.id2name: Dict[int, str] = {}
        try:
            with open(classes_yaml, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            names = y.get("names")
            if isinstance(names, dict):
                for k, v in names.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    self.id2name[idx] = str(v)
            elif isinstance(names, list):
                for idx, v in enumerate(names):
                    self.id2name[idx] = str(v)
        except Exception as e:
            print(f"[WARN] Could not read classes.yaml: {e}", file=sys.stderr)

    def id_to_name(self, idx: int) -> str:
        """Return the class name for a given index (fallback 'class_{idx}')."""
        return self.id2name.get(idx, f"class_{idx}")


class ImageSet:
    """
    Discovers image files under a given IMAGES_DIR.

    Methods
    -------
    list_images() -> List[Path]
        Returns sorted list of image paths with supported extensions.
    """
    def __init__(self, images_dir: Path):
        self.images_dir = images_dir

    def list_images(self) -> List[Path]:
        """Return all images with supported extensions under IMAGES_DIR (non-recursive)."""
        files: List[Path] = []
        for ext in IMAGE_EXTS:
            files.extend(sorted(self.images_dir.glob(f"*{ext}")))
        return sorted(files)


class Detector:
    """
    Thin wrapper around Ultralytics YOLO for inference.

    Methods
    -------
    detect(img_path: Path) -> List[Detection]
        Run the model on an image file and return detections in normalized xywh.
    """
    def __init__(self, weights_path: Path):
        """
        Parameters
        ----------
        weights_path : Path
            Path to ultralytics .pt weights.
        """
        self.enabled = _HAS_YOLO and weights_path.exists()
        self.model = None
        if self.enabled:
            try:
                self.model = YOLO(str(weights_path))
            except Exception as e:
                self.enabled = False
                print(f"[WARN] Failed to load YOLO model: {e}", file=sys.stderr)

    def detect(self, img_path: Path) -> List[Detection]:
        """
        Run detection and return a list of Detection objects.

        Parameters
        ----------
        img_path : Path

        Returns
        -------
        List[Detection]
        """
        if not self.enabled or self.model is None:
            return []

        results = self.model.predict(source=str(img_path), verbose=False)
        if not results:
            return []

        r = results[0]
        dets: List[Detection] = []
        try:
            xywhn = r.boxes.xywhn.cpu().numpy() if hasattr(r.boxes, "xywhn") else None
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else None
            clss = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else None
            if xywhn is None or confs is None or clss is None:
                return dets
            for i in range(xywhn.shape[0]):
                cx, cy, w, h = map(float, xywhn[i])
                conf = float(confs[i])
                cls_id = int(clss[i])
                dets.append(Detection(cls_id=cls_id, conf=conf, cx=cx, cy=cy, w=w, h=h))
        except Exception as e:
            print(f"[WARN] Could not parse detections for {img_path.name}: {e}", file=sys.stderr)
        return dets


class App(tk.Tk):
    """
    Main GUI application for detection review and label repair.

    UI Overview
    -----------
    Left:   Image canvas with drawn bounding boxes and mouse-edit handles.
    Right:  Controls + list:
            - Subset selector (train/val/test)
            - Image navigation (Prev/Next, index)
            - Run Detect, Load Labels, Save
            - Treeview listing detections [cls, conf, cx, cy, w, h, class_name]
              (delete from here)

    Mouse Editing
    -------------
    - Click a box to select it.
    - Drag INSIDE the selected box to move it.
    - Drag a HANDLE to resize (8 handles: corners & edges).
    - Press Delete to remove the selected detection.

    Keyboard
    --------
    Left/Right: previous/next image
    Ctrl+S    : save labels
    Delete    : delete selected detection
    """

    def __init__(self):
        super().__init__()
        self.title("YOLO Detect & Label Repair GUI")
        self.geometry("1280x800")

        # State
        self.subset_var = tk.StringVar(value="test")  # train/val/test
        self.current_images_dir = IMAGES_DIR
        self.current_labels_dir = LABELS_DIR
        self.class_map = ClassMap(CLASSES_YAML)
        self.image_set = ImageSet(self.current_images_dir)
        self.images: List[Path] = self.image_set.list_images()
        self.idx = 0  # current image index
        self.label_store = LabelStore(self.current_labels_dir)
        self.detector = Detector(MODEL_PATH)

        # Working detections for the currently displayed image
        self.working: List[Detection] = []

        # Canvas / image drawing state
        self.canvas: tk.Canvas
        self.canvas_img = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.img_w = 1
        self.img_h = 1
        self.last_pil: Optional[Image.Image] = None  # keep original for re-scaling

        # Mouse interaction state
        self.selected_idx: Optional[int] = None
        self.drag_mode: Optional[str] = None  # None | 'move' | 'resize'
        self.active_handle: Optional[str] = None  # 'nw','n','ne','e','se','s','sw','w'
        self.drag_start_xy: Tuple[int, int] = (0, 0)
        self.handle_radius_px = 6  # half-size of handle squares

        # UI
        self._build_layout()
        self._load_current_image_and_labels()

        # Shortcuts
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Control-s>", lambda e: self.save_labels())
        self.bind("<Delete>", lambda e: self.delete_selected_detection())

    # ----------------------------- UI Construction ---------------------------------

    def _build_layout(self) -> None:
        """
        Build and place all UI components: canvas, controls, treeview (list), and buttons.
        """
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: image canvas
        left = ttk.Frame(main)
        main.add(left, weight=3)

        self.canvas = tk.Canvas(left, bg="#222222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw_canvas())
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Right: control + list panel
        right = ttk.Frame(main, padding=10)
        main.add(right, weight=2)

        # Subset selector
        subset_row = ttk.Frame(right)
        subset_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(subset_row, text="Subset:").pack(side=tk.LEFT)
        subset_cb = ttk.Combobox(subset_row, textvariable=self.subset_var,
                                 values=["train", "val", "test"], state="readonly", width=8)
        subset_cb.pack(side=tk.LEFT, padx=6)
        subset_cb.bind("<<ComboboxSelected>>", lambda e: self._on_subset_change())

        # Navigation row
        nav = ttk.Frame(right)
        nav.pack(fill=tk.X, pady=(0, 8))
        self.idx_var = tk.StringVar(value="0 / 0")
        ttk.Button(nav, text="◀ Prev", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next ▶", command=self.next_image).pack(side=tk.LEFT, padx=6)
        ttk.Label(nav, textvariable=self.idx_var).pack(side=tk.LEFT, padx=10)

        # Actions row
        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text="Run Detect", command=self.run_detect).pack(side=tk.LEFT)
        ttk.Button(actions, text="Load Labels", command=self.load_labels_only).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Labels", command=self.save_labels).pack(side=tk.LEFT, padx=6)

        # Treeview list of detections (no in-form editing, just selection + delete)
        cols = ("cls_id", "conf", "cx", "cy", "w", "h", "class_name")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for c, w in zip(cols, (60, 70, 70, 70, 70, 70, 140)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(8, 6))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._on_tree_select())

        # Delete button
        del_row = ttk.Frame(right)
        del_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(del_row, text="Delete Selected", command=self.delete_selected_detection).pack(side=tk.LEFT)

        # Hint
        hint = ttk.Label(
            right,
            text="Mouse: click box to select, drag to move.\nDrag handles to resize. Delete key removes selection.",
            foreground="#555"
        )
        hint.pack(fill=tk.X, pady=(6, 0))

    # ----------------------------- Data Loading -------------------------------------

    def _on_subset_change(self) -> None:
        """
        Handle subset (train/val/test) selection change and reload file list.
        """
        subset = self.subset_var.get()
        base = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset")
        self.current_images_dir = base / "images" / subset
        self.current_labels_dir = base / "labels" / subset
        self.image_set = ImageSet(self.current_images_dir)
        self.images = self.image_set.list_images()
        self.label_store = LabelStore(self.current_labels_dir)
        self.idx = 0
        if not self.images:
            messagebox.showwarning("No Images", f"No images found in:\n{self.current_images_dir}")
        self._load_current_image_and_labels()

    def _load_current_image_and_labels(self) -> None:
        """
        Load current image (by index), draw it, and load labels into working list & table.
        """
        count = len(self.images)
        self.idx_var.set(f"{self.idx+1 if count else 0} / {count}")
        if not self.images:
            self.canvas.delete("all")
            self._clear_tree()
            self.selected_idx = None
            return

        img_path = self.images[self.idx]
        self._open_image(img_path)
        self.working = self.label_store.load_for_image(img_path)
        self.selected_idx = None
        self._refresh_tree()
        self._redraw_canvas()

    def _open_image(self, img_path: Path) -> None:
        """
        Open the image (store original PIL) and draw sized version in the canvas.

        Parameters
        ----------
        img_path : Path
        """
        try:
            self.last_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Image Load Error", f"Could not open {img_path.name}:\n{e}")
            self.last_pil = None
            return

        self._fit_and_draw_pil(self.last_pil)

    def _fit_and_draw_pil(self, pil: Image.Image) -> None:
        """
        Fit PIL image to current canvas size while preserving aspect ratio, then draw.

        Parameters
        ----------
        pil : Image.Image
        """
        c_w = self.canvas.winfo_width() or 800
        c_h = self.canvas.winfo_height() or 600
        img_w, img_h = pil.size

        scale = min(c_w / max(1, img_w), c_h / max(1, img_h))
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        self.img_w, self.img_h = new_w, new_h
        pil_resized = pil.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_resized)

        self.canvas.delete("all")
        self.canvas_img = self.canvas.create_image(
            (c_w // 2, c_h // 2),
            image=self.tk_img,
            anchor=tk.CENTER
        )

    # ----------------------------- Navigation ---------------------------------------

    def prev_image(self) -> None:
        """Go to previous image (wrapping)."""
        if not self.images:
            return
        self.idx = (self.idx - 1) % len(self.images)
        self._load_current_image_and_labels()

    def next_image(self) -> None:
        """Go to next image (wrapping)."""
        if not self.images:
            return
        self.idx = (self.idx + 1) % len(self.images)
        self._load_current_image_and_labels()

    # ----------------------------- Detection & Labels -------------------------------

    def run_detect(self) -> None:
        """
        Run the detector on the current image and replace the working list with detections.
        """
        if not self.images:
            return
        if not self.detector.enabled:
            messagebox.showwarning("Detector Unavailable", "Ultralytics YOLO not available or best.pt missing.")
            return
        img_path = self.images[self.idx]
        dets = self.detector.detect(img_path)
        self.working = dets
        self.selected_idx = None
        self._refresh_tree()
        self._redraw_canvas()

    def load_labels_only(self) -> None:
        """
        Load labels (.txt) from disk for the current image and set as working list.
        """
        if not self.images:
            return
        img_path = self.images[self.idx]
        self.working = self.label_store.load_for_image(img_path)
        self.selected_idx = None
        self._refresh_tree()
        self._redraw_canvas()

    def save_labels(self) -> None:
        """
        Save the working list to the label .txt for the current image (overwrite).
        """
        if not self.images:
            return
        img_path = self.images[self.idx]
        try:
            self.label_store.save_for_image(img_path, self.working)
            messagebox.showinfo("Saved", f"Labels saved for {img_path.name}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save labels:\n{e}")

    # ----------------------------- Tree (List) --------------------------------------

    def _clear_tree(self) -> None:
        """Remove all rows from the treeview."""
        for iid in self.tree.get_children():
            self.tree.delete(iid)

    def _refresh_tree(self) -> None:
        """
        Re-populate the tree from 'working' detections.
        """
        self._clear_tree()
        for i, d in enumerate(self.working):
            cname = self.class_map.id_to_name(d.cls_id)
            self.tree.insert(
                "", "end", iid=str(i),
                values=(d.cls_id, f"{d.conf:.3f}", f"{d.cx:.4f}", f"{d.cy:.4f}",
                        f"{d.w:.4f}", f"{d.h:.4f}", cname)
            )
        # Reselect if possible
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.working):
            self.tree.selection_set(str(self.selected_idx))
        else:
            self.tree.selection_remove(self.tree.selection())

    def _on_tree_select(self) -> None:
        """
        When a row is selected in the list, reflect selection on canvas.
        """
        sel = self.tree.selection()
        if not sel:
            self.selected_idx = None
        else:
            self.selected_idx = int(sel[0])
        self._redraw_canvas()

    def delete_selected_detection(self) -> None:
        """
        Delete the currently selected detection (from list or canvas).
        """
        if self.selected_idx is None:
            # if list has selection, try that
            sel = self.tree.selection()
            if sel:
                self.selected_idx = int(sel[0])
            else:
                return
        if 0 <= self.selected_idx < len(self.working):
            del self.working[self.selected_idx]
            self.selected_idx = None
            self._refresh_tree()
            self._redraw_canvas()

    # ----------------------------- Mouse Editing ------------------------------------

    def _on_canvas_click(self, event: tk.Event) -> None:
        """
        Handle mouse click: select box; decide move vs resize depending on handle proximity.

        Parameters
        ----------
        event : tk.Event (Button-1)
        """
        if not self.working:
            return

        # Determine which box is under cursor (prefer last drawn / latest index for intuitive overlap)
        clicked_idx = self._hit_test_box(event.x, event.y)
        if clicked_idx is None:
            # Clicked empty space: clear selection
            self.selected_idx = None
            self.drag_mode = None
            self.active_handle = None
            self._refresh_tree()
            self._redraw_canvas()
            return

        self.selected_idx = clicked_idx
        self._refresh_tree()
        self._redraw_canvas()

        # If on a handle -> resize; else if inside -> move
        handle = self._hit_test_handle(event.x, event.y, clicked_idx)
        if handle is not None:
            self.drag_mode = "resize"
            self.active_handle = handle
        else:
            # Inside the box area
            self.drag_mode = "move"
            self.active_handle = None

        self.drag_start_xy = (event.x, event.y)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        """
        Handle mouse drag: move/resize selected detection.

        Parameters
        ----------
        event : tk.Event (B1-Motion)
        """
        if self.selected_idx is None or not (0 <= self.selected_idx < len(self.working)):
            return
        if self.drag_mode is None:
            return

        d = self.working[self.selected_idx]

        # Convert canvas XY to normalized coords helpers
        cx_px, cy_px, half_w_px, half_h_px = self._xywhn_to_center_half_px(d.cx, d.cy, d.w, d.h)

        if self.drag_mode == "move":
            # Compute delta in pixels
            dx = event.x - self.drag_start_xy[0]
            dy = event.y - self.drag_start_xy[1]
            # New pixel center
            new_cx_px = cx_px + dx
            new_cy_px = cy_px + dy
            # Convert back to normalized
            new_cx_n, new_cy_n = self._canvas_px_to_norm(new_cx_px, new_cy_px)
            d.cx = max(0.0, min(1.0, new_cx_n))
            d.cy = max(0.0, min(1.0, new_cy_n))

        elif self.drag_mode == "resize":
            # Figure current xyxy in pixels, then adjust the side we're dragging
            x0, y0, x1, y1 = self._xywhn_to_xyxy_pixels(d.cx, d.cy, d.w, d.h)
            # Normalize to ensure x0<=x1, y0<=y1
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0

            # Update sides based on active handle
            x, y = event.x, event.y
            handle = self.active_handle  # 'nw','n','ne','e','se','s','sw','w'
            if handle in ("nw", "w", "sw"):
                x0 = x
            if handle in ("ne", "e", "se"):
                x1 = x
            if handle in ("nw", "n", "ne"):
                y0 = y
            if handle in ("sw", "s", "se"):
                y1 = y

            # Prevent inverted boxes
            min_w = 2  # pixels
            min_h = 2
            if x1 - x0 < min_w:
                # Lock to min width
                if handle in ("nw", "w", "sw"):
                    x0 = x1 - min_w
                else:
                    x1 = x0 + min_w
            if y1 - y0 < min_h:
                if handle in ("nw", "n", "ne"):
                    y0 = y1 - min_h
                else:
                    y1 = y0 + min_h

            # Convert xyxy back to normalized cx,cy,w,h
            cxn0, cyn0 = self._canvas_px_to_norm((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            wn0 = max(0.0, min(1.0, (x1 - x0) / max(1, self.img_w)))
            hn0 = max(0.0, min(1.0, (y1 - y0) / max(1, self.img_h)))
            d.cx = max(0.0, min(1.0, cxn0))
            d.cy = max(0.0, min(1.0, cyn0))
            d.w = wn0
            d.h = hn0

        self.drag_start_xy = (event.x, event.y)
        # Live redraw + list update
        self._refresh_tree()
        self._redraw_canvas()

    def _on_canvas_release(self, event: tk.Event) -> None:
        """
        End a drag operation.
        """
        self.drag_mode = None
        self.active_handle = None

    # ----------------------------- Drawing ------------------------------------------

    def _redraw_canvas(self) -> None:
        """
        Redraw the image and all bounding boxes (and handles for current selection).
        """
        if self.last_pil is None:
            return
        self._fit_and_draw_pil(self.last_pil)

        # Draw boxes
        for i, d in enumerate(self.working):
            x0, y0, x1, y1 = self._xywhn_to_xyxy_pixels(d.cx, d.cy, d.w, d.h)
            outline = "#00FF00" if i != self.selected_idx else "#00FFFF"
            width = 2 if i != self.selected_idx else 3
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=width)

            # Label tag
            cname = self.class_map.id_to_name(d.cls_id)
            self.canvas.create_text(x0 + 4, y0 + 8, text=f"{cname} ({d.conf:.2f})",
                                    anchor=tk.NW, fill=outline, font=("TkDefaultFont", 9, "bold"))

        # Draw handles for selected box
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.working):
            d = self.working[self.selected_idx]
            handles = self._get_handle_positions(d)
            for hx, hy in handles.values():
                self._draw_handle(hx, hy)

    def _draw_handle(self, x: int, y: int) -> None:
        """
        Draw a square handle centered at (x,y).

        Parameters
        ----------
        x, y : int
            Canvas pixel coordinates.
        """
        r = self.handle_radius_px
        self.canvas.create_rectangle(x - r, y - r, x + r, y + r,
                                     outline="#FFFFFF", fill="#00AEEF", width=1)

    # ----------------------------- Hit Testing & Geometry ---------------------------

    def _get_image_canvas_offset(self) -> Tuple[int, int]:
        """
        Compute top-left (left, top) of the drawn image inside the canvas.

        Returns
        -------
        (img_left, img_top) : Tuple[int, int]
        """
        c_w = self.canvas.winfo_width() or self.img_w
        c_h = self.canvas.winfo_height() or self.img_h
        img_left = (c_w - self.img_w) // 2
        img_top = (c_h - self.img_h) // 2
        return img_left, img_top

    def _xywhn_to_center_half_px(self, cx: float, cy: float, w: float, h: float) -> Tuple[int, int, float, float]:
        """
        Convert normalized xywh to center pixel + half sizes in pixels (no canvas offset).

        Parameters
        ----------
        cx, cy, w, h : float

        Returns
        -------
        (cx_px, cy_px, half_w_px, half_h_px)
        """
        cx_px = cx * self.img_w
        cy_px = cy * self.img_h
        half_w_px = 0.5 * w * self.img_w
        half_h_px = 0.5 * h * self.img_h
        return int(round(cx_px)), int(round(cy_px)), half_w_px, half_h_px

    def _xywhn_to_xyxy_pixels(self, cx: float, cy: float, w: float, h: float) -> Tuple[int, int, int, int]:
        """
        Convert normalized xywh to integer pixel xyxy for the current displayed image size.

        Parameters
        ----------
        cx, cy, w, h : float

        Returns
        -------
        (x0, y0, x1, y1) : Tuple[int, int, int, int]
            Top-left and bottom-right pixel coords within the canvas.
        """
        cx_px, cy_px, half_w_px, half_h_px = self._xywhn_to_center_half_px(cx, cy, w, h)
        x0 = int(round(cx_px - half_w_px))
        y0 = int(round(cy_px - half_h_px))
        x1 = int(round(cx_px + half_w_px))
        y1 = int(round(cy_px + half_h_px))
        img_left, img_top = self._get_image_canvas_offset()
        return x0 + img_left, y0 + img_top, x1 + img_left, y1 + img_top

    def _canvas_px_to_norm(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert canvas pixel coords back to normalized (cx, cy) relative to the *displayed* image.

        Parameters
        ----------
        x, y : float
            Canvas pixel coordinates.

        Returns
        -------
        (cx_norm, cy_norm) : Tuple[float, float]
        """
        img_left, img_top = self._get_image_canvas_offset()
        # Shift into image-local coords
        xi = x - img_left
        yi = y - img_top
        # Clamp within image
        xi = max(0.0, min(float(self.img_w), xi))
        yi = max(0.0, min(float(self.img_h), yi))
        return xi / max(1, self.img_w), yi / max(1, self.img_h)

    def _hit_test_box(self, x: int, y: int) -> Optional[int]:
        """
        Return the index of the topmost box under canvas pixel (x,y), else None.

        Parameters
        ----------
        x, y : int

        Returns
        -------
        Optional[int]
        """
        # Iterate from last to first so visually topmost gets priority
        for i in reversed(range(len(self.working))):
            d = self.working[i]
            x0, y0, x1, y1 = self._xywhn_to_xyxy_pixels(d.cx, d.cy, d.w, d.h)
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return None

    def _get_handle_positions(self, d: Detection) -> Dict[str, Tuple[int, int]]:
        """
        Get canvas pixel positions of the 8 resize handles for detection d.

        Returns
        -------
        Dict[str, (x,y)] with keys: 'nw','n','ne','e','se','s','sw','w'
        """
        x0, y0, x1, y1 = self._xywhn_to_xyxy_pixels(d.cx, d.cy, d.w, d.h)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        return {
            "nw": (x0, y0),
            "n":  (cx, y0),
            "ne": (x1, y0),
            "e":  (x1, cy),
            "se": (x1, y1),
            "s":  (cx, y1),
            "sw": (x0, y1),
            "w":  (x0, cy),
        }

    def _hit_test_handle(self, x: int, y: int, idx: int) -> Optional[str]:
        """
        If (x,y) is over a resize handle for detection at idx, return that handle key.

        Parameters
        ----------
        x, y : int
        idx : int

        Returns
        -------
        Optional[str]
        """
        if not (0 <= idx < len(self.working)):
            return None
        d = self.working[idx]
        handles = self._get_handle_positions(d)
        r = self.handle_radius_px + 2  # detection tolerance
        for key, (hx, hy) in handles.items():
            if abs(x - hx) <= r and abs(y - hy) <= r:
                return key
        return None

    # ------------------------------------ main --------------------------------------

    def _redraw_and_sync(self) -> None:
        """Utility: refresh both tree and canvas."""
        self._refresh_tree()
        self._redraw_canvas()


# ------------------------------------ script main -----------------------------------

def main() -> None:
    """
    Entrypoint: builds and runs the Tkinter application.
    """
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
