import os
from PIL import Image

img_dir = "data/weapons/train/images"
lbl_dir = "data/weapons/train/labels"

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
    if not os.path.exists(lbl_path):
        print("MISSING LABEL:", img_name)
