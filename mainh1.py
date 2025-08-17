import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
from ultralytics import YOLO
import cvzone
import os
from datetime import datetime

# -------------------------------
# Load YOLO model (helmet detection)
# -------------------------------
model = YOLO('best.pt')  # Make sure this model is trained for Helmet vs No Helmet

# -------------------------------
# Folder setup
# -------------------------------
folder_name = "HelmetDetections"
os.makedirs(folder_name, exist_ok=True)
log_file = os.path.join(folder_name, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# -------------------------------
# Globals
# -------------------------------
video_path = None
esp32_url = None


# -------------------------------
# Detection Function
# -------------------------------
def run_detection():
    global video_path, esp32_url
    source = esp32_url if esp32_url else video_path
    cap = cv2.VideoCapture(source)

    frame_count = 0
    fps = 0
    prev_time = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # FPS Calculation
        now_time = datetime.now()
        elapsed = (now_time - prev_time).total_seconds()
        fps = 1 / elapsed if elapsed > 0 else 0
        prev_time = now_time

        # Run YOLO
        results = model.predict(frame, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().numpy()

            for (box, class_id, conf) in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                label = model.names[class_id]
                confidence = float(conf)

                # Draw detection
                color = (0, 255, 0) if label.lower() == "helmet" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    scale=1,
                    thickness=2,
                    colorT=(255, 255, 255),
                    colorR=color,
                    offset=5
                )

                # Save detections with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file, "a") as f:
                    f.write(f"{timestamp} | {label} | Conf: {confidence:.2f}\n")

        # Show FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Tkinter Functions
# -------------------------------
def select_video():
    global video_path, esp32_url
    esp32_url = None  # reset
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if video_path:
        status_label.config(text=f"Video Selected: {os.path.basename(video_path)}")


def use_esp32():
    global esp32_url, video_path
    video_path = None  # reset
    # Ask user for ESP32 stream URL
    esp32_url = url_entry.get().strip()
    if esp32_url:
        status_label.config(text=f"ESP32-CAM URL Set: {esp32_url}")
    else:
        messagebox.showerror("Error", "Please enter ESP32-CAM stream URL.")


def start_detection():
    if not video_path and not esp32_url:
        messagebox.showerror("No Source", "Please select a video or enter ESP32 URL first.")
        return
    threading.Thread(target=run_detection, daemon=True).start()


# -------------------------------
# Tkinter GUI
# -------------------------------
root = tk.Tk()
root.title("YOLOv8 Helmet Detection")
root.geometry("700x400")
root.configure(bg="#1e1e1e")

tk.Label(root, text="YOLOv8 Helmet Detection", fg="white", bg="#1e1e1e",
         font=("Helvetica", 20, "bold")).pack(pady=20)

tk.Button(root, text="Select Video", command=select_video,
          font=("Helvetica", 14), bg="gray", fg="white", width=20).pack(pady=10)

# ESP32 input
tk.Label(root, text="ESP32-CAM Stream URL:", fg="white", bg="#1e1e1e",
         font=("Helvetica", 12)).pack(pady=5)
url_entry = tk.Entry(root, width=50, font=("Helvetica", 12))
url_entry.insert(0, "http://192.168.0.103/stream")  # placeholder, usually port 81
url_entry.pack(pady=5)
tk.Button(root, text="Use ESP32-CAM", command=use_esp32,
          font=("Helvetica", 14), bg="orange", fg="black", width=20).pack(pady=10)

tk.Button(root, text="Run Detection", command=start_detection,
          font=("Helvetica", 14), bg="#007acc", fg="white", width=20).pack(pady=10)

status_label = tk.Label(root, text="No source selected", fg="white", bg="#1e1e1e")
status_label.pack(pady=10)

root.mainloop()
