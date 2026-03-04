# 🚗 Detecting Distracted Drivers using CNN

A real-time driver distraction detection system built using Deep Learning and Computer Vision.  
The system classifies driver activities using a MobileNetV2-based CNN model and triggers an alert when distracted behavior persists.

---

## 🎯 Problem

Driver distraction is a major cause of road accidents.  
This project builds a vision-based monitoring system that detects unsafe behaviors using only a dashboard camera.

---

## 🧠 Model

- Architecture: MobileNetV2 (Transfer Learning)
- Input Size: 224x224
- Output Classes: 10 driver activities
- Loss: Categorical Crossentropy
- Optimizer: Adam

### Final Model Performance
- Accuracy: 97.70%
- Precision: 97.72%
- Recall: 97.70%
- F1-Score: 97.69%

---

## 🚨 Features

- Real-time webcam detection (OpenCV)
- GUI interface (Tkinter)
- Frame-based prediction
- Alert triggered after 3 seconds of distraction
- Lightweight and deployable design

---

## 🛠 Tech Stack

Python, TensorFlow, Keras, OpenCV, Tkinter, NumPy, Keras Tuner

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py

```markdown
## 📸 Output Screenshots

![Safe Driving]
![Picture5](https://github.com/user-attachments/assets/2eab28c4-a03d-49b6-89ed-16f467d1ea1f)

![Distracted Alert]
![Picture2](https://github.com/user-attachments/assets/fcc58090-cda3-4c65-ae69-2f0152ef41e5)
![Picture7](https://github.com/user-attachments/assets/e5c61b91-0cb9-4de5-9404-2e64b7f8ed5d)



