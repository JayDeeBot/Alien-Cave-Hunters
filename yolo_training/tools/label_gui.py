#!/usr/bin/env python3
"""
YOLO Label Maker GUI (Tkinter)
- Iterates images in a folder
- Loads classes from YAML
- Single-class selection via exclusive checkboxes
- Multiple bounding boxes per image (drag to draw; press D to delete last; C to clear all)
- Saves YOLO txt labels (class_id cx cy w h), normalized to [0,1]
- Supports negative images (creates an EMPTY .txt)

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
# IMAGES_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/images/train") # Uncomment for training set 
# LABELS_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/labels/train") # use 80/20 split training/validation

IMAGES_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/images/val") # Uncomment for validation set
LABELS_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/labels/val")

# IMAGES_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/images/test") # Uncomment for testing set
# LABELS_DIR = Path("/home/jarred/git/Alien-Cave-Hunters/yolo_training/dataset/labels/test")

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


def yolo_save_label_lines(label_path: Path, lines: List[Tuple[int, float, float, float, float]]):
    """
    Save multiple YOLO label lines.
    Each entry: (class_id, cx_n, cy_n, w_n, h_n) already normalized.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cid, cx, cy, w, h in lines:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def yolo_load_label_lines(label_path: Path) -> Optional[List[Tuple[int, float, float, float, float]]]:
    """
    Load all YOLO label lines.
    Returns list of (class_id, cx, cy, w, h) normalized, or:
      - [] if file exists but is empty (negative image)
      - None if file does not exist or parsing fails
    """
    if not label_path.exists():
        return None
    try:
        with open(label_path, "r") as f:
            content = f.read().strip()
        if content == "":
            return []  # Explicit negative
        lines = []
        for line in content.splitlines():
            parts = line.split()
            if len(parts) != 5:
                return None
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            lines.append((cid, cx, cy, w, h))
        return lines
    except Exception:
        return None


def yolo_save_empty_label(label_path: Path):
    """
    Creates an empty YOLO label file for negative images (no objects).
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w"):
        pass


def yolo_is_empty_label(label_path: Path) -> bool:
    """
    Returns True if the label file exists and is empty (negative image).
    """
    try:
        return label_path.exists() and label_path.stat().st_size == 0
    except Exception:
        return False


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

        # Canvas-drawing state
        self.dragging = False
        self.start_xy = (0, 0)
        self.end_xy = (0, 0)
        self.temp_bbox_canvas = None  # transient rectangle while dragging

        # Multi-box storage: each item is dict with
        # {"cid": int, "xyxy_img": (x1,y1,x2,y2), "rect_id": int}
        self.boxes: List[dict] = []

        # Selected class (exclusive checkboxes behavior)
        self.selected_class_id: Optional[int] = None
        self.class_vars: List[tk.IntVar] = [tk.IntVar(value=0) for _ in self.class_names]

        # Negative-image toggle (no artefact / empty label)
        self.negative_var = tk.IntVar(value=0)

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

        self.class_checkbuttons: List[ttk.Checkbutton] = []
        for i, name in enumerate(self.class_names):
            cb = ttk.Checkbutton(
                classes_frame,
                text=name,
                variable=self.class_vars[i],
                command=lambda i=i: self.on_class_toggle(i)
            )
            cb.pack(anchor="w", padx=8, pady=2)
            self.class_checkbuttons.append(cb)

        # Negative (no-artefact) toggle
        neg_frame = ttk.LabelFrame(right, text="Negative Image")
        neg_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.neg_cb = ttk.Checkbutton(
            neg_frame,
            text="No artefact present (create empty label)",
            variable=self.negative_var,
            command=self.on_negative_toggle
        )
        self.neg_cb.pack(anchor="w", padx=8, pady=4)

        # Buttons
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        self.btn_save = ttk.Button(btn_frame, text="Save (S)", command=self.save_label)
        self.btn_save.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.btn_clear = ttk.Button(btn_frame, text="Clear All Boxes (C)", command=self.clear_all_boxes)
        self.btn_clear.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        self.btn_del_last = ttk.Button(btn_frame, text="Delete Last (D)", command=self.delete_last_box)
        self.btn_del_last.grid(row=1, column=0, padx=4, pady=4, sticky="ew")

        self.btn_prev = ttk.Button(btn_frame, text="Prev (P)", command=self.prev_image)
        self.btn_prev.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        self.btn_next = ttk.Button(btn_frame, text="Next (N)", command=self.next_image)
        self.btn_next.grid(row=2, column=0, padx=4, pady=4, sticky="ew")

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
        self.master.bind("<Key-c>", lambda e: self.clear_all_boxes())
        self.master.bind("<Key-C>", lambda e: self.clear_all_boxes())
        self.master.bind("<Key-d>", lambda e: self.delete_last_box())
        self.master.bind("<Key-D>", lambda e: self.delete_last_box())
        # Toggle negative label
        self.master.bind("<Key-z>", lambda e: self.neg_cb.invoke())
        self.master.bind("<Key-Z>", lambda e: self.neg_cb.invoke())

    # ---------------------- Class selection logic ----------------------
    def on_class_toggle(self, i: int):
        """Exclusive behavior: only one checkbox can be active at a time."""
        # If negative is enabled, ignore class toggles
        if self.negative_var.get() == 1:
            self.class_vars[i].set(0)
            return

        if self.class_vars[i].get() == 1:
            for j, var in enumerate(self.class_vars):
                if j != i:
                    var.set(0)
            self.selected_class_id = i
        else:
            self.selected_class_id = None

    def set_selected_class(self, class_id: Optional[int]):
        for i, var in enumerate(self.class_vars):
            var.set(1 if class_id is not None and i == class_id else 0)
        self.selected_class_id = class_id

    def on_negative_toggle(self):
        """
        When 'No artefact present' is toggled:
        - If ON: clear temp bbox, disable classes, block drawing, and clear any existing boxes.
        - If OFF: re-enable class selection and drawing.
        """
        if self.negative_var.get() == 1:
            self.set_selected_class(None)
            self.clear_all_boxes()
            self._clear_temp_rect()
            for cb in self.class_checkbuttons:
                cb.state(["disabled"])
            self.status.config(text="Negative mode: will save an EMPTY label file.")
        else:
            for cb in self.class_checkbuttons:
                cb.state(["!disabled"])
            self.status.config(text="")

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
        self._clear_temp_rect()
        self.boxes.clear()  # clear all persistent boxes
        self.canvas.create_image((MAX_CANVAS_W // 2, MAX_CANVAS_H // 2), image=self.tk_img, anchor="center")

        # Centering offsets for drawing and mapping canvas→image
        self.offset_x = (MAX_CANVAS_W - disp_w) // 2
        self.offset_y = (MAX_CANVAS_H - disp_h) // 2

        # Reset negative toggle and classes by default
        self.negative_var.set(0)
        self.on_negative_toggle()  # ensures UI state (enables classes)

        # Show info
        self.info_lbl.config(text=f"Image {self.idx+1}/{len(self.images)}\n{img_path.name}  ({W}×{H})")
        self.status.config(text="")

        # Load labels (multiple or empty)
        lbl_path = self.label_path_for(img_path)
        lines = yolo_load_label_lines(lbl_path)
        if lines is None:
            # No label present
            return
        if len(lines) == 0:
            # Empty = negative
            self.negative_var.set(1)
            self.on_negative_toggle()
            self.status.config(text="Loaded existing EMPTY label (negative).")
            return

        # Recreate all boxes
        for cid, cx, cy, w, h in lines:
            x1 = (cx - w / 2.0) * W
            y1 = (cy - h / 2.0) * H
            x2 = (cx + w / 2.0) * W
            y2 = (cy + h / 2.0) * H
            c1x = self.offset_x + int(x1 * self.scale_x)
            c1y = self.offset_y + int(y1 * self.scale_y)
            c2x = self.offset_x + int(x2 * self.scale_x)
            c2y = self.offset_y + int(y2 * self.scale_y)
            rect_id = self._draw_persistent_rect(c1x, c1y, c2x, c2y)
            self.boxes.append({"cid": cid, "xyxy_img": (x1, y1, x2, y2), "rect_id": rect_id})
        self.status.config(text=f"Loaded {len(self.boxes)} existing box(es).")

    def label_path_for(self, img_path: Path) -> Path:
        return LABELS_DIR / (img_path.stem + ".txt")

    # ---------------------- BBox drawing helpers ----------------------
    def _draw_persistent_rect(self, x1c: int, y1c: int, x2c: int, y2c: int) -> int:
        """Draw a rectangle that represents a saved box (green)."""
        return self.canvas.create_rectangle(x1c, y1c, x2c, y2c, outline="#00ff00", width=2)

    def _draw_or_update_temp_rect(self, p1: Tuple[int,int], p2: Tuple[int,int]):
        """Draw/update a temporary rectangle while dragging (cyan)."""
        x1, y1 = p1
        x2, y2 = p2
        x1 = max(0, min(x1, MAX_CANVAS_W))
        x2 = max(0, min(x2, MAX_CANVAS_W))
        y1 = max(0, min(y1, MAX_CANVAS_H))
        y2 = max(0, min(y2, MAX_CANVAS_H))
        if self.temp_bbox_canvas is None:
            self.temp_bbox_canvas = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ffff", width=2, dash=(4, 2))
        else:
            self.canvas.coords(self.temp_bbox_canvas, x1, y1, x2, y2)

    def _clear_temp_rect(self):
        if self.temp_bbox_canvas is not None:
            self.canvas.delete(self.temp_bbox_canvas)
            self.temp_bbox_canvas = None

    # ---------------------- Mouse events ----------------------
    def on_press(self, event):
        if not self.tk_img or self.negative_var.get() == 1:
            return
        # Must have a class selected before drawing
        if self.selected_class_id is None:
            self.status.config(text="Select a class before drawing a box.")
            return
        self.dragging = True
        self.start_xy = (event.x, event.y)
        self.end_xy = (event.x, event.y)
        self._draw_or_update_temp_rect(self.start_xy, self.end_xy)

    def on_drag(self, event):
        if not self.dragging:
            return
        if self.negative_var.get() == 1:
            return
        self.end_xy = (event.x, event.y)
        self._draw_or_update_temp_rect(self.start_xy, self.end_xy)

    def on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        if self.negative_var.get() == 1:
            return
        self.end_xy = (event.x, event.y)
        self._draw_or_update_temp_rect(self.start_xy, self.end_xy)

        # Finalize box: convert canvas coords to original image coords and store
        x1c, y1c = self.start_xy
        x2c, y2c = self.end_xy
        x1c, x2c = min(x1c, x2c), max(x1c, x2c)
        y1c, y2c = min(y1c, y2c), max(y1c, y2c)

        # Ignore tiny/zero-size boxes
        if abs(x2c - x1c) < 2 or abs(y2c - y1c) < 2:
            self._clear_temp_rect()
            return

        # convert canvas -> original image coordinates
        W, H = self.orig_img.size
        x1i = (x1c - self.offset_x) / self.scale_x
        y1i = (y1c - self.offset_y) / self.scale_y
        x2i = (x2c - self.offset_x) / self.scale_x
        y2i = (y2c - self.offset_y) / self.scale_y

        # Clamp to original image bounds
        x1i = max(0, min(x1i, W - 1))
        x2i = max(1, min(x2i, W))
        y1i = max(0, min(y1i, H - 1))
        y2i = max(1, min(y2i, H))

        # Persist the rectangle (green) and clear the temp cyan one
        rect_id = self._draw_persistent_rect(x1c, y1c, x2c, y2c)
        self._clear_temp_rect()

        # Store persistent box
        self.boxes.append({"cid": self.selected_class_id, "xyxy_img": (x1i, y1i, x2i, y2i), "rect_id": rect_id})
        self.status.config(text=f"Added box #{len(self.boxes)} (class={self.selected_class_id}).")

    # ---------------------- Box management ----------------------
    def clear_all_boxes(self):
        """Remove all persistent boxes from canvas and memory."""
        for b in self.boxes:
            try:
                self.canvas.delete(b["rect_id"])
            except Exception:
                pass
        self.boxes.clear()
        self.status.config(text="Cleared all boxes.")

    def delete_last_box(self):
        """Remove the most recently added box (LIFO)."""
        if not self.boxes:
            self.status.config(text="No boxes to delete.")
            return
        last = self.boxes.pop()
        try:
            self.canvas.delete(last["rect_id"])
        except Exception:
            pass
        self.status.config(text=f"Deleted last box. {len(self.boxes)} remaining.")

    # ---------------------- Navigation & Save ----------------------
    def prev_image(self):
        if self.idx > 0:
            self.load_image(self.idx - 1)

    def next_image(self):
        if self.idx < len(self.images) - 1:
            self.load_image(self.idx + 1)

    def save_label(self):
        img_path = self.images[self.idx]
        label_path = self.label_path_for(img_path)

        # Negative: save empty file and advance.
        if self.negative_var.get() == 1:
            yolo_save_empty_label(label_path)
            self.status.config(text=f"Saved EMPTY label: {label_path.name}")
            if self.idx < len(self.images) - 1:
                self.load_image(self.idx + 1)
            return

        # Must have at least one box
        if not self.boxes:
            messagebox.showwarning("Missing boxes", "Please draw at least one bounding box or toggle Negative.")
            return

        # Build label lines (normalized)
        W, H = self.orig_img.size
        lines = []
        for b in self.boxes:
            cid = b["cid"]
            if cid is None:
                messagebox.showwarning("Missing class", "One or more boxes have no class selected.")
                return
            x1, y1, x2, y2 = b["xyxy_img"]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            cx_n = cx / W
            cy_n = cy / H
            w_n = w / W
            h_n = h / H
            lines.append((cid, cx_n, cy_n, w_n, h_n))

        yolo_save_label_lines(label_path, lines)
        self.status.config(text=f"Saved {len(lines)} box(es): {label_path.name}")

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
