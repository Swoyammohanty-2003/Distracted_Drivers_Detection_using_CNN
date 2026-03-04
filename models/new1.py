import cv2
import numpy as np
import tensorflow as tf
import winsound
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import time

# Load final trained model (make sure the file exists in the same directory)
model = tf.keras.models.load_model("best_driver_detection_model.keras")

# Class labels
classes_dict = {
    0: "Safe Driving",
    1: "Texting Right",
    2: "Talking on Phone - Right",
    3: "Texting Left",
    4: "Talking on Phone - Left",
    5: "Operating Radio",
    6: "Drinking",
    7: "Reaching Behind",
    8: "Hair & Makeup",
    9: "Talking to Passenger"
}

# Constants
IMG_SIZE = 224
FRAME_THRESHOLD = 20  # Number of consecutive distracted frames before alert
frame_count = 0
cap = None
running = False

# UI setup
root = tk.Tk()
root.title("Driver Distraction Detection")
root.geometry("800x600")

label_text = tk.StringVar()
label_text.set("Status: Waiting...")

video_label = Label(root)
video_label.pack()

status_label = Label(root, textvariable=label_text, font=("Arial", 14), fg="red")
status_label.pack()

def start_detection():
    global cap, running, frame_count
    running = True
    frame_count = 0
    cap = cv2.VideoCapture(0)
    process_video()

def stop_detection():
    global running, cap
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    label_text.set("Status: Stopped")

def process_video():
    global frame_count

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        label_text.set("Error: No frame from webcam.")
        return

    # Preprocess image for model
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    img_array = np.expand_dims(normalized, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(prediction))
    label = classes_dict[predicted_class]

    # Alert handling
    if predicted_class != 0:
        frame_count += 1
    else:
        frame_count = 0

    if frame_count >= FRAME_THRESHOLD:
        label_text.set("ALERT! Driver Distracted!")
        winsound.Beep(1000, 500)
        frame_count = 0
    else:
        label_text.set(f"Status: {label}")

    # Display video
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(img)
    video_label.configure(image=imgtk)
    video_label.image = imgtk

    root.after(10, process_video)

# Buttons
start_button = Button(root, text="Start Detection", command=lambda: threading.Thread(target=start_detection, daemon=True).start(), font=("Arial", 12))
start_button.pack(pady=5)

stop_button = Button(root, text="Stop Detection", command=stop_detection, font=("Arial", 12))
stop_button.pack(pady=5)

root.mainloop()
