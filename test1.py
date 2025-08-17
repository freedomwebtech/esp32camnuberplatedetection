import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cvzone
import os
from datetime import datetime

# Load YOLO and OCR
model = YOLO('best.pt')  # Trained to detect license plates
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en'
)

# Folder setup
folder_name = "DetectedPlates"
os.makedirs(folder_name, exist_ok=True)
log_file = os.path.join(folder_name, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Globals
video_path = None
esp32_url = None
saved_ids = set()
id_to_plate = {}

# --- Detection Function ---
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
        frame = cv2.resize(frame, (1020, 600))
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        # FPS Calculation
        now_time = datetime.now()
        elapsed = (now_time - prev_time).total_seconds()
        fps = 1 / elapsed if elapsed > 0 else 0
        prev_time = now_time

        results = model.track(frame, persist=True)
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for track_id, box, class_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                label = model.names[class_id]

                if label == "license plate":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cvzone.putTextRect(frame, f"LICENSE PLATE", (x1, y1 - 10),
                                       scale=1, thickness=2,
                                       colorT=(255, 255, 255), colorR=(0, 0, 255), offset=5)

                    cropped = frame[y1:y2, x1:x2]

                    if track_id not in id_to_plate:
                        # --- PaddleOCR with predict() ---
                        result = ocr.predict(cropped)

                        plate_text = ""
                        if result and isinstance(result, list) and "rec_texts" in result[0]:
                            rec_texts = result[0]["rec_texts"]
                            plate_text = " ".join(rec_texts).strip()
                            print(f"Full Plate: {plate_text}")

                        if plate_text:
                            id_to_plate[track_id] = plate_text
                            if track_id not in saved_ids:
                                saved_ids.add(track_id)
                                # Save cropped image
                                cv2.imwrite(os.path.join(folder_name, f"plate_{track_id}.jpg"), cropped)
                                # Save log
                                with open(log_file, 'a') as f:
                                    f.write(f"{datetime.now()} | ID: {track_id} | Plate: {plate_text}\n")
                                print(f"[ALERT] New Plate Detected → ID: {track_id} | Plate: {plate_text}")

                    # Always show ID
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x1, y2 + 10),
                                       scale=1, thickness=2,
                                       colorT=(0, 0, 0), colorR=(0, 255, 255), offset=3)

        # Show FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# --- UI Setup ---
def select_video():
    global video_path, esp32_url
    esp32_url = None  # reset
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
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
    threading.Thread(target=run_detection).start()

root = tk.Tk()
root.title("YOLOv8 License Plate Detection")
root.geometry("700x400")
root.configure(bg="#1e1e1e")

tk.Label(root, text="YOLOv8 Video/ESP32 Detection", fg="white", bg="#1e1e1e",
         font=("Helvetica", 20, "bold")).pack(pady=20)

tk.Button(root, text="Select Video", command=select_video,
          font=("Helvetica", 14), bg="gray", fg="white", width=20).pack(pady=10)

# ESP32 input
tk.Label(root, text="ESP32-CAM Stream URL:", fg="white", bg="#1e1e1e",
         font=("Helvetica", 12)).pack(pady=5)
url_entry = tk.Entry(root, width=50, font=("Helvetica", 12))
url_entry.insert(0, "http://192.168.0.103/stream")  # placeholder
url_entry.pack(pady=5)
tk.Button(root, text="Use ESP32-CAM", command=use_esp32,
          font=("Helvetica", 14), bg="orange", fg="black", width=20).pack(pady=10)

tk.Button(root, text="Run Detection", command=start_detection,
          font=("Helvetica", 14), bg="#007acc", fg="white", width=20).pack(pady=10)

status_label = tk.Label(root, text="No source selected", fg="white", bg="#1e1e1e")
status_label.pack(pady=10)

root.mainloop()
