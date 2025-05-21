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
- A USB or Pi Camera Module
- One LED connected to GPIO pin

## ⚙️ Installation

```bash
git clone https://github.com/emrullahalmsns/raspberry-pi-crack-detector.git
cd raspberry-pi-crack-detector
pip install -r requirements.txt

