#!/usr/bin/env python3
import cv2
import os

# input and output directories
input_dir = '/home/millie_ral/caveexplorer_ws/src/Alien-Cave-Hunters/yolo_training/dataset/images/train'
output_dir = '/home/millie_ral/caveexplorer_ws/src/Alien-Cave-Hunters/yolo_training/dataset/images/degraded'

os.makedirs(output_dir, exist_ok=True)

# get list of images
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for idx, filename in enumerate(image_files):
    path_in = os.path.join(input_dir, filename)
    img = cv2.imread(path_in)
    if img is None:
        print(f"⚠️  Could not read {path_in}, skipping.")
        continue

    # # 1) Gaussian blur
    # gaus = cv2.GaussianBlur(img, (7,7), 0)
    # gaus_name = f"gaus-{filename}"
    # cv2.imwrite(os.path.join(output_dir, gaus_name), gaus)
    # print(f"saved Gaussian-blurred: {gaus_name}")

    # # 2) Bilateral filter
    # bilat = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # bilat_name = f"bilat-{filename}"
    # cv2.imwrite(os.path.join(output_dir, bilat_name), bilat)
    # print(f"saved Bilateral-filtered: {bilat_name}")

    # 3) Median blur
    median = cv2.medianBlur(img, ksize=5)  # kernel size must be odd
    median_name = f"median-{filename}"
    cv2.imwrite(os.path.join(output_dir, median_name), median)
    print(f"saved Median-blurred: {median_name}")


print(f"Done! Degraded images saved to {output_dir}")
