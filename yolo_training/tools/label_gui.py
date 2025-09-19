#!/usr/bin/env python3
"""
YOLO Label Maker GUI (Tkinter)
- Iterates images in a folder
- Loads classes from YAML
- Single-class selection via exclusive checkboxes
- Single bounding box per image (drag to draw/adjust)
- Saves YOLO txt labels (class_id cx cy w h), normalized to [0,1]

Paths are set to your project structure.
"""

import os
import sys
import glob
import math
import yaml
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk  # pip install pillow

# ---------------------- CONFIG: update if you change repo layout ----------------------
IMAGES_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/images/train")
LABELS_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/labels/train")
CLASSES_YAML = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/tools/classes.yaml")

# Max canvas size for display (image will be scaled to fit; coordinates are unscaled on save)
MAX_CANVAS_W = 1280
MAX_CANVAS_H = 800
# -------------------------------------------------------------------------------------


def load_class_names(yaml_path: Path) -> List[str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Classes file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("Invalid classes.yaml format. Expected:\n\nnames:\n  - class1\n  - class2\n")
    return names


def list_images(images_dir: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(sorted(images_dir.glob(ext)))
    return files


def yolo_save_label(label_path: Path, class_id: int, bbox_xyxy: Tuple[float, float, float, float], img_size: Tuple[int, int]):
    """
    bbox_xyxy: (x1, y1, x2, y2) in *original image coordinates*
    img_size: (W, H)
    """
    W, H = img_size
    x1, y1, x2, y2 = bbox_xyxy

    # Clamp to image bounds
    x1 = max(0, min(x1, W - 1))
    x2 = max(1, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(1, min(y2, H))

    # Ensure proper ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    # Normalize
    cx_n = cx / W
    cy_n = cy / H
    w_n = w / W
    h_n = h / H

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")


def yolo_load_label(label_path: Path) -> Optional[Tuple[int, Tuple[float, float, float, float]]]:
    """
    Returns (class_id, (cx, cy, w, h)) normalized, or None if not exists.
    """
    if not label_path.exists():
        return None
    try:
        with open(label_path, "r") as f:
            line = f.readline().strip()
        parts = line.split()
        if len(parts) != 5:
            return None
        cid = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        return cid, (cx, cy, w, h)
    except Exception:
        return None


class LabelGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("YOLO Label Maker")
        master.geometry(f"{MAX_CANVAS_W+360}x{MAX_CANVAS_H+80}")  # window size: canvas + side panel

        # Data
        self.class_names = load_class_names(CLASSES_YAML)
        self.images = list_images(IMAGES_DIR)
        if not self.images:
            messagebox.showerror("No images", f"No images found in {IMAGES_DIR}")
            sys.exit(1)

        self.idx = 0
        self.orig_img: Optional[Image.Image] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.current_bbox_canvas = None  # canvas rectangle id
        self.dragging = False
        self.start_xy = (0, 0)
        self.end_xy = (0, 0)

        # Selected class (exclusive checkboxes behavior)
        self.selected_class_id: Optional[int] = None
        self.class_vars: List[tk.IntVar] = [tk.IntVar(value=0) for _ in self.class_names]

        # Layout
        self._build_widgets()
        self._bind_keys()
        self.load_image(self.idx)

    # ---------------------- UI construction ----------------------
    def _build_widgets(self):
        # Left: image canvas
        left = ttk.Frame(self.master)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, bg="#111", width=MAX_CANVAS_W, height=MAX_CANVAS_H, cursor="tcross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Mouse bindings for bbox
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Right: controls
        right = ttk.Frame(self.master)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # File info
        self.info_lbl = ttk.Label(right, text="", justify=tk.LEFT)
        self.info_lbl.pack(anchor="w", padx=8, pady=(10, 6))

        # Classes (as exclusive checkboxes)
        classes_frame = ttk.LabelFrame(right, text="Classes")
        classes_frame.pack(fill=tk.X, padx=8, pady=8)

        for i, name in enumerate(self.class_names):
            cb = ttk.Checkbutton(
                classes_frame,
                text=name,
                variable=self.class_vars[i],
                command=lambda i=i: self.on_class_toggle(i)
            )
            cb.pack(anchor="w", padx=8, pady=2)

        # Buttons
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        self.btn_save = ttk.Button(btn_frame, text="Save (S)", command=self.save_label)
        self.btn_save.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.btn_clear = ttk.Button(btn_frame, text="Clear Box (C)", command=self.clear_bbox)
        self.btn_clear.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        self.btn_prev = ttk.Button(btn_frame, text="Prev (P)", command=self.prev_image)
        self.btn_prev.grid(row=1, column=0, padx=4, pady=4, sticky="ew")

        self.btn_next = ttk.Button(btn_frame, text="Next (N)", command=self.next_image)
        self.btn_next.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        # Status
        self.status = ttk.Label(right, text="", foreground="#08a", justify=tk.LEFT)
        self.status.pack(anchor="w", padx=8, pady=(6, 10))

    def _bind_keys(self):
        self.master.bind("<Key-s>", lambda e: self.save_label())
        self.master.bind("<Key-S>", lambda e: self.save_label())
        self.master.bind("<Key-n>", lambda e: self.next_image())
        self.master.bind("<Key-N>", lambda e: self.next_image())
        self.master.bind("<Key-p>", lambda e: self.prev_image())
        self.master.bind("<Key-P>", lambda e: self.prev_image())
        self.master.bind("<Key-c>", lambda e: self.clear_bbox())
        self.master.bind("<Key-C>", lambda e: self.clear_bbox())

    # ---------------------- Class selection logic ----------------------
    def on_class_toggle(self, i: int):
        """Exclusive behavior: only one checkbox can be active at a time."""
        if self.class_vars[i].get() == 1:
            # turn off all others
            for j, var in enumerate(self.class_vars):
                if j != i:
                    var.set(0)
            self.selected_class_id = i
        else:
            # user unchecked the selected one
            self.selected_class_id = None

    def set_selected_class(self, class_id: Optional[int]):
        for i, var in enumerate(self.class_vars):
            var.set(1 if class_id is not None and i == class_id else 0)
        self.selected_class_id = class_id

    # ---------------------- Image loading & scaling ----------------------
    def load_image(self, idx: int):
        idx = max(0, min(idx, len(self.images) - 1))
        self.idx = idx
        img_path = self.images[self.idx]
        self.orig_img = Image.open(img_path).convert("RGB")
        W, H = self.orig_img.size

        # Fit to canvas while preserving aspect
        scale = min(MAX_CANVAS_W / W, MAX_CANVAS_H / H, 1.0)
        disp_w = int(W * scale)
        disp_h = int(H * scale)
        self.scale_x = scale
        self.scale_y = scale

        disp_img = self.orig_img.resize((disp_w, disp_h), Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(disp_img)

        self.canvas.delete("all")
        self.current_bbox_canvas = None
        self.canvas.create_image((MAX_CANVAS_W // 2, MAX_CANVAS_H // 2), image=self.tk_img, anchor="center")

        # Centering offsets for drawing and mapping canvas→image
        self.offset_x = (MAX_CANVAS_W - disp_w) // 2
        self.offset_y = (MAX_CANVAS_H - disp_h) // 2

        # Show info
        self.info_lbl.config(text=f"Image {self.idx+1}/{len(self.images)}\n{img_path.name}  ({W}×{H})")
        self.status.config(text="")

        # If label exists, load and draw it
        lbl_path = self.label_path_for(img_path)
        loaded = yolo_load_label(lbl_path)
        if loaded:
            cid, (cx, cy, w, h) = loaded
            # convert normalized to original xyxy
            x1 = (cx - w / 2.0) * W
            y1 = (cy - h / 2.0) * H
            x2 = (cx + w / 2.0) * W
            y2 = (cy + h / 2.0) * H
            # map to canvas coords
            c1x = self.offset_x + int(x1 * self.scale_x)
            c1y = self.offset_y + int(y1 * self.scale_y)
            c2x = self.offset_x + int(x2 * self.scale_x)
            c2y = self.offset_y + int(y2 * self.scale_y)
            self.draw_or_update_bbox((c1x, c1y), (c2x, c2y))
            self.set_selected_class(cid)
            self.status.config(text=f"Loaded existing label: class={cid}")

    def label_path_for(self, img_path: Path) -> Path:
        return LABELS_DIR / (img_path.stem + ".txt")

    # ---------------------- BBox drawing ----------------------
    def on_press(self, event):
        if not self.tk_img:
            return
        self.dragging = True
        self.start_xy = (event.x, event.y)
        self.end_xy = (event.x, event.y)
        self.draw_or_update_bbox(self.start_xy, self.end_xy)

    def on_drag(self, event):
        if not self.dragging:
            return
        self.end_xy = (event.x, event.y)
        self.draw_or_update_bbox(self.start_xy, self.end_xy)

    def on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        self.end_xy = (event.x, event.y)
        self.draw_or_update_bbox(self.start_xy, self.end_xy)

    def draw_or_update_bbox(self, p1: Tuple[int,int], p2: Tuple[int,int]):
        x1, y1 = p1
        x2, y2 = p2
        # keep within canvas
        x1 = max(0, min(x1, MAX_CANVAS_W))
        x2 = max(0, min(x2, MAX_CANVAS_W))
        y1 = max(0, min(y1, MAX_CANVAS_H))
        y2 = max(0, min(y2, MAX_CANVAS_H))
        if self.current_bbox_canvas is None:
            self.current_bbox_canvas = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff00", width=2)
        else:
            self.canvas.coords(self.current_bbox_canvas, x1, y1, x2, y2)

    def clear_bbox(self):
        if self.current_bbox_canvas is not None:
            self.canvas.delete(self.current_bbox_canvas)
            self.current_bbox_canvas = None
        self.status.config(text="Cleared bounding box.")

    # ---------------------- Navigation & Save ----------------------
    def prev_image(self):
        if self.idx > 0:
            self.load_image(self.idx - 1)

    def next_image(self):
        if self.idx < len(self.images) - 1:
            self.load_image(self.idx + 1)

    def save_label(self):
        if self.selected_class_id is None:
            messagebox.showwarning("Missing class", "Please select a class before saving.")
            return
        if self.current_bbox_canvas is None:
            messagebox.showwarning("Missing box", "Please draw a bounding box before saving.")
            return

        coords = self.canvas.coords(self.current_bbox_canvas)  # [x1,y1,x2,y2] in canvas coords
        x1c, y1c, x2c, y2c = coords
        # ensure order
        x1c, x2c = min(x1c, x2c), max(x1c, x2c)
        y1c, y2c = min(y1c, y2c), max(y1c, y2c)

        # convert canvas -> original image coordinates
        x1i = (x1c - self.offset_x) / self.scale_x
        y1i = (y1c - self.offset_y) / self.scale_y
        x2i = (x2c - self.offset_x) / self.scale_x
        y2i = (y2c - self.offset_y) / self.scale_y

        # Clamp to original image bounds
        W, H = self.orig_img.size
        x1i = max(0, min(x1i, W - 1))
        x2i = max(1, min(x2i, W))
        y1i = max(0, min(y1i, H - 1))
        y2i = max(1, min(y2i, H))

        img_path = self.images[self.idx]
        label_path = self.label_path_for(img_path)

        yolo_save_label(label_path, self.selected_class_id, (x1i, y1i, x2i, y2i), (W, H))
        self.status.config(text=f"Saved: {label_path.name}")
        # Auto-advance
        if self.idx < len(self.images) - 1:
            self.load_image(self.idx + 1)

# ---------------------- Main ----------------------
def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    try:
        style = ttk.Style()
        # Use a nice theme if available
        style.theme_use("clam")
    except Exception:
        pass
    app = LabelGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
