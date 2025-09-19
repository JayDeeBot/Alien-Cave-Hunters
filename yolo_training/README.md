# How to train the yolo agent

## Classes
```yaml
names:
  - stop_sign
  - rock
  - panel
  - crate
```

## Labelling Template /.txt
```php
<class_id> <cx> <cy> <w> <h>   # all normalized to [0,1]
```

## Organising Images & Labels

1. Collect images and add them to /dataset/images/train/*.jpg|png
2. Add labels for each image to /dataset/labels/train/*.txt
3. Use the labelling gui for quick labelling


## Label GUI
```bash
python 3 tools/label_gui.py
```

## Check Labels

```bash
# 1: unpaired files (train)
comm -3 \
 <(ls dataset/images/train | sed 's/\.[^.]*$//' | sort) \
 <(ls dataset/labels/train | sed 's/\.[^.]*$//' | sort) \
 | sed 's/^/UNPAIRED: /'

# 2: spot a few label lines for range/format
head -n 3 dataset/labels/train/*.txt | sed -n '1,10p'
```

## Train Yolo Agent

yolo task=detect mode=train \
  model=yolov8n.pt \
  data=dataset/data.yaml \
  epochs=50 imgsz=640 batch=16 \
  optimizer=auto seed=42 patience=20 \
  project=runs_cave name=yolov8n_baseline

## Export & version the artefact

cp runs_cave/yolov8n_baseline/weights/best.pt \
   ../cave_explorer/config/best.pt

