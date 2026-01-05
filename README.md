# Car_License_Plate_Detection

🚗 License Plate Detection Using YOLO11 & EasyOCR
An end-to-end Automatic Number Plate Recognition (ANPR) system using YOLO11 for detection and EasyOCR for text extraction. The system is trained on a custom dataset and supports both training and inference.

Ideal for smart surveillance, parking systems, and automated toll collection.



🔧 Features

Convert PASCAL VOC XML annotations to YOLO format
Split dataset into train/test sets
Train YOLO11m model for license plate detection
Perform inference on images
Crop detected license plate regions
Recognize text using EasyOCR
Full pipeline from raw data to license plate text output

🛠️ Technologies Used 

Python 3.x
Ultralytics YOLO11 – State-of-the-art object detection
EasyOCR – Optical Character Recognition (OCR)
OpenCV (cv2) – Image processing
NumPy, Matplotlib, Glob, XML, YAML
Google Colab / Local GPU – For training
WANDB (optional) – Experiment tracking


📦 Installation
1. Clone the Repository
Bash

git clone https://github.com/jagan007G/Car_License_Plate_Detection.git
cd license-plate-detection

2. Create Virtual Environment (Recommended)
Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

📁 Project Structure

license-plate-detection-yolo11/

│

├── data/

│   ├── annotations/           # PASCAL VOC XML files (Cars0.xml, Cars1.xml, ...)

│   └── images/                 # Input images (Cars0.png, Cars1.png, ...)

│

├── License-Plate-Data/        # Generated dataset

│   ├── train/

│   │   ├── images/

│   │   └── labels/

│   ├── test/

│   │   ├── images/

│   │   └── labels/

│   └── data.yaml               # YOLO dataset config

│

├── runs/                       # Training outputs (weights, logs)

│

├── predict.py                  # Inference script (optional)

├── train.py                    # Training script (optional)

├── colab_notebook.ipynb      # Full notebook (this one)

├── README.md

└── requirements.txt

🖼️ Dataset Requirements
Images in data/images/ (e.g., Cars0.png, Cars1.png)
Corresponding XML annotations in data/annotations/ (e.g., Cars0.xml)
XML format: PASCAL VOC (bounding box with <object><bndbox>)
License plate must be labeled as class 0 in XML
