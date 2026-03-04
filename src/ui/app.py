import cv2
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

from src.preprocessing.image_preprocessing import preprocess_frame
from src.prediction.predictor import Predictor
from src.alert.alert_system import AlertSystem
from src.config.config import FRAME_THRESHOLD

def run_app():
    root = tk.Tk()
    root.title("Driver Distraction Detection")
    root.geometry("800x600")

    label_text = tk.StringVar()
    label_text.set("Status: Waiting...")

    video_label = Label(root)
    video_label.pack()

    status_label = Label(root, textvariable=label_text, font=("Arial", 14), fg="red")
    status_label.pack()

    predictor = Predictor("models/best_driver_detection_model.keras")
    alert_system = AlertSystem(FRAME_THRESHOLD)

    cap = None
    running = False

    def start_detection():
        nonlocal cap, running
        running = True
        cap = cv2.VideoCapture(0)
        process_video()

    def stop_detection():
        nonlocal running, cap
        running = False
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        label_text.set("Status: Stopped")

    def process_video():
        nonlocal cap, running
        if not running or cap is None:
            return

        ret, frame = cap.read()
        if not ret:
            label_text.set("Error: No frame from webcam.")
            return

        img_array = preprocess_frame(frame)
        predicted_class, label = predictor.predict(img_array)

        if alert_system.check_alert(predicted_class):
            label_text.set("ALERT! Driver Distracted!")
        else:
            label_text.set(f"Status: {label}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        video_label.configure(image=imgtk)
        video_label.image = imgtk

        root.after(10, process_video)

    start_button = Button(root, text="Start Detection",
                          command=lambda: threading.Thread(target=start_detection, daemon=True).start(),
                          font=("Arial", 12))
    start_button.pack(pady=5)

    stop_button = Button(root, text="Stop Detection",
                         command=stop_detection,
                         font=("Arial", 12))
    stop_button.pack(pady=5)

    root.mainloop()
