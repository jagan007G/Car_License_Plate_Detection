# 🚗 Automatic Number Plate Recognition (ANPR) using YOLO11 and EasyOCR

An end-to-end Automatic Number Plate Recognition (ANPR) system that detects vehicle license plates using **YOLO11** and recognizes the license plate text using **EasyOCR**.

The project demonstrates the complete deep learning workflow, including dataset preprocessing, annotation conversion, model training, object detection, and Optical Character Recognition (OCR).

---

# 📖 Overview

Automatic Number Plate Recognition (ANPR) is widely used in intelligent transportation systems, smart parking, toll collection, traffic monitoring, and vehicle access control.

This project implements the complete ANPR pipeline:

- Prepare a custom dataset
- Convert PASCAL VOC XML annotations into YOLO format
- Split the dataset into training and validation sets
- Train a custom YOLO11 object detection model
- Detect license plates from vehicle images
- Crop detected license plates
- Extract license plate text using EasyOCR

---

# ✨ Features

- Custom dataset preprocessing
- PASCAL VOC XML to YOLO annotation conversion
- Automatic dataset splitting (80% Train / 20% Validation)
- YOLO11m model training
- License plate detection
- Automatic license plate cropping
- Optical Character Recognition (OCR) using EasyOCR
- End-to-end Automatic Number Plate Recognition pipeline

---

# 🛠️ Technologies Used

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Object Detection | Ultralytics YOLO11 |
| OCR | EasyOCR |
| Computer Vision | OpenCV |
| Data Processing | NumPy, XML, YAML |
| Visualization | Matplotlib |
| Deep Learning | PyTorch (Ultralytics Backend) |
| Development Environment | Google Colab |

---

# 📂 Repository Structure

```
Car_License_Plate_Detection/
│
├── images/
│   └── Vehicle Images
│
├── annotations/
│   └── PASCAL VOC XML Files
│
├── License-Plate-Data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── data.yaml
│
├── ANPR.ipynb
│
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/jagan007G/Car_License_Plate_Detection.git

cd Car_License_Plate_Detection
```

Install the required packages

```bash
pip install ultralytics
pip install easyocr
pip install opencv-python
pip install matplotlib
pip install numpy
```

---

# 📊 Dataset Preparation

The dataset consists of

- Vehicle Images (.png)
- PASCAL VOC XML annotation files

The notebook automatically

- Reads XML annotations
- Converts bounding boxes into YOLO format
- Creates the YOLO dataset structure
- Splits the dataset into

```
80% Training

20% Validation
```

A `data.yaml` configuration file is generated automatically for YOLO training.

---

# 🚀 Model Training

The project trains the **YOLO11m** object detection model.

```python
model = YOLO("yolo11m.pt")

model.train(
    data="data.yaml",
    epochs=70,
    imgsz=640
)
```

Training outputs are stored in

```
runs/detect/train/
```

The trained model is saved as

```
best.pt
```

---

# 🔍 License Plate Detection

The trained YOLO model predicts the bounding box coordinates of the license plate.

Example workflow

```
Vehicle Image

↓

YOLO11 Detection

↓

Bounding Box Coordinates

↓

Crop License Plate

↓

EasyOCR

↓

License Plate Number
```

---

# 🔠 Optical Character Recognition

EasyOCR extracts the characters from the cropped license plate image.

Example

```
Detected Plate

TN09AB1234
```

---

# 🔄 Complete Pipeline

```
Vehicle Image
      │
      ▼
Read XML Annotation
      │
      ▼
Convert XML → YOLO Format
      │
      ▼
Generate YOLO Dataset
      │
      ▼
Train YOLO11
      │
      ▼
Load Trained Model
      │
      ▼
Detect License Plate
      │
      ▼
Crop Plate Region
      │
      ▼
EasyOCR
      │
      ▼
Recognized License Plate
```

---

# 📸 Results

## Input Image

<img width="1024" height="768" alt="001" src="https://github.com/user-attachments/assets/7189497d-a1c6-42bf-98b8-7370470e8d41" />
<img width="1200" height="900" alt="bike" src="https://github.com/user-attachments/assets/51db880c-cf55-4aa8-a0d2-67bf2db3b999" />
<img width="768" height="1024" alt="number_plate" src="https://github.com/user-attachments/assets/69c976fd-4689-4310-ba86-2caf450397c8" />
<img width="1400" height="500" alt="number_plate2" src="https://github.com/user-attachments/assets/23fefe46-8e6b-4dd6-bbc3-b522cf3db943" />
![Uploading number_plate4.jpg…]()


---

## License Plate Detection

<img width="697" height="628" alt="detected" src="https://github.com/user-attachments/assets/4e69a34d-0942-4a9c-8eea-29943af197ca" />

---

## Cropped License Plate

<img width="117" height="67" alt="Screenshot 2026-07-19 153447" src="https://github.com/user-attachments/assets/bb41bde8-c03e-43f3-8457-f0fb0f8d443a" />

<img width="116" height="68" alt="Screenshot 2026-07-19 153432" src="https://github.com/user-attachments/assets/6b30b22e-9d4b-4ddf-af6b-14198030a827" />



---

## OCR Result

```
MH23 DV2363
```

# 💼 Applications

- Smart Parking Systems
- Automatic Toll Collection
- Vehicle Access Control
- Traffic Surveillance
- Intelligent Transportation Systems
- Smart Cities
- Law Enforcement

---

# 🚀 Future Improvements

- Real-time video processing
- Webcam integration
- Multiple vehicle detection
- Vehicle tracking
- OCR confidence filtering
- Flask/FastAPI deployment
- Streamlit Web Application
- Docker support
- Edge deployment using NVIDIA Jetson


# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
