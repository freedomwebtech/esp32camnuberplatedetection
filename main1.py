import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

# Load model and class list
model = YOLO('best.pt')
with open("coco1.txt", "r") as f:
    class_list = f.read().split("\n")

video_path = None  # Global variable to hold video source

# Detection logic in a thread
def run_detection():
    global video_path
    if not video_path:
        messagebox.showerror("Error", "No video source selected.")
        return

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showinfo("Info", "Video playback completed or stream ended.")
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for _, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = class_list[d]

            color = (0, 0, 255) if 'accident' in c else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# UI setup
def browse_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if video_path:
        status_label.config(text=f"Selected: {os.path.basename(video_path)}", fg="lightgreen")
    else:
        status_label.config(text="No file selected", fg="red")

def use_esp32cam():
    global video_path
    video_path = "http://192.168.0.103/stream"
    status_label.config(text="ESP32-CAM Live Stream Selected", fg="lightblue")

def start_detection_thread():
    threading.Thread(target=run_detection, daemon=True).start()

# Build the tkinter UI
root = tk.Tk()
root.title("YOLOv8 Crash Detection")
root.geometry("500x300")
root.configure(bg="#1e1e1e")

title = tk.Label(root, text="YOLOv8 Video Detection", font=("Arial", 16), bg="#1e1e1e", fg="white")
title.pack(pady=20)

browse_btn = tk.Button(root, text="Select Video File", command=browse_video, bg="#333", fg="white", padx=10, pady=5)
browse_btn.pack(pady=10)

esp32_btn = tk.Button(root, text="Use ESP32-CAM Stream", command=use_esp32cam, bg="#444", fg="white", padx=10, pady=5)
esp32_btn.pack(pady=10)

run_btn = tk.Button(root, text="Run Detection", command=start_detection_thread, bg="#007acc", fg="white", padx=10, pady=5)
run_btn.pack(pady=10)

status_label = tk.Label(root, text="No video selected", font=("Arial", 10), bg="#1e1e1e", fg="gray")
status_label.pack(pady=20)

root.mainloop()
