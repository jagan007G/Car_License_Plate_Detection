pip install ultralytics -q

pip install easyocr

import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import glob
import random
import shutil
import yaml
from ultralytics import YOLO
import easyocr

ann_file = '/content/drive/MyDrive/Colab_Notebooks/annotations/Cars0.xml'
image_file = '/content/drive/MyDrive/Colab_Notebooks/images/Cars0.png'

def process_annotations(ann_file):
    tree = ET.parse(ann_file)
    root = tree.getroot()
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)
    return width, height, xmin, ymin, xmax, ymax

def bounding_box_in_yolo_format(width, height, xmin, ymin, xmax, ymax):
    xmin_new = xmin / width
    xmax_new = xmax / width
    ymin_new = ymin / height
    ymax_new = ymax / height
    width_new = xmax_new - xmin_new
    height_new = ymax_new - ymin_new
    x = xmin_new + (width_new / 2)
    y = ymin_new + (height_new / 2)
    return x, y, width_new, height_new

w, h, xmin, ymin, xmax, ymax = process_annotations(ann_file)
print(process_annotations(ann_file))

image = cv2.imread(image_file)

cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.axis(False)
plt.show()

bounding_box_in_yolo_format(*process_annotations(ann_file))

all_images = glob.glob('/content/drive/MyDrive/Colab_Notebooks/images/*.png')
all_annotations = glob.glob('/content/drive/MyDrive/Colab_Notebooks/annotations/*.xml')

os.mkdir('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data')

os.mkdir('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/train')
os.mkdir('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/train/images')
os.mkdir('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/train/labels')

os.mkdir('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/test')
os.mkdir('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/test/images')
os.mkdir('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/test/labels')

total = list(range(len(all_images)))
random.shuffle(total)

train_split = 0.8
train_size = int(len(all_images) * train_split)
test_size = int(len(all_images)) - train_size

train_images = total[:train_size]
test_images = total[train_size:]

for i in sorted(train_images):
    shutil.copy(f'/content/drive/MyDrive/Colab_Notebooks/images/Cars{i}.png', '/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/train/images/')
    x, y, w, h = bounding_box_in_yolo_format(
        *process_annotations(f'/content/drive/MyDrive/Colab_Notebooks/annotations/Cars{i}.xml')
    )
    with open(f'/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/train/labels/Cars{i}.txt', 'w') as f:
        f.write(f"0 {x} {y} {w} {h}")

for i in sorted(test_images):
    shutil.copy(f'/content/drive/MyDrive/Colab_Notebooks/images/Cars{i}.png', '/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/test/images/')
    x, y, w, h = bounding_box_in_yolo_format(
        *process_annotations(f'/content/drive/MyDrive/Colab_Notebooks/annotations/Cars{i}.xml')
    )
    with open(f'/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/test/labels/Cars{i}.txt', 'w') as f:
        f.write(f"0 {x} {y} {w} {h}")

data = dict(
    train = '/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/train',
    val = '/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/test',
    nc = 1,
    names = {0: 'license_plate'}
)

with open('/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/data.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

!wandb off

model = YOLO('yolo11m.pt')

from ultralytics import settings

settings.update({'datasets_dir': '/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/'})
print(settings)

results = model.train(data='/content/drive/MyDrive/Colab_Notebooks/License-Plate-Data/data.yaml', epochs=70, imgsz=640)

model = YOLO('/content/runs/detect/train/weights/best.pt')

def predict_box(image):
    bbox = []
    results = model.predict(image)
    for res in results:
        boxes = res.boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
            bbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    return bbox[0]

def crop_image(bbox, image):
    xmin, ymin, xmax, ymax = bbox
    new_image = image[ymin:ymax, xmin:xmax]
    return new_image

reader = easyocr.Reader(['en'])
img_file = '/content/drive/MyDrive/Colab_Notebooks/test_image/bike.jpg'

img = cv2.imread(img_file)
bbox = predict_box(img)
image = crop_image(bbox, img)
plt.imshow(image)
results = reader.readtext(image)
print(results)
l = []
for res in results:
    l.append(res[1])
print(f"License Plate: {''.join(l)}")