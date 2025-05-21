# Raspberry Pi Crack Detector 🔍

This project is a **real-time crack detection system** running on **Raspberry Pi**, powered by **YOLOv11** object detection. When a crack is detected in the camera feed, an **LED** is triggered via GPIO.

## 📸 Features

- Runs on Raspberry Pi (tested on Pi 5)
- Real-time crack detection with YOLOv11
- Activates LED when a crack is found
- Python-based, easy to deploy

## 🛠️ Requirements

- Raspberry Pi OS (Bullseye or newer)
- Python 3.8+
- `ultralytics` (YOLOv11)
- `opencv-python`
- `torch`
- `ncnn`
- A USB or Pi Camera Module
- One LED connected to GPIO17 (physical pin 11) and GND

## ⚙️ Installation & Setup

```bash

sudo apt update && sudo apt upgrade -y

mkdir yolo
cd yolo

python -m venv --system-site-packages venv
source venv/bin/activate

pip install ultralytics ncnn opencv-python torch

ls /dev/video*   # To check if the camera is detected

# Export YOLO model to NCNN format (if needed)
yolo export model=my_model.pt format=ncnn

# Run detection script with model and camera input
python yolo_detect.py --model=my_model_ncnn_model --source=usb0 --resolution=1280x720
