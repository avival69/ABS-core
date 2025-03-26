import cv2
import torch
from ultralytics import YOLO
from flask import Flask, jsonify
import threading

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8x").to(device)

# Price list
price_list = {
    "bottle": 100,
    "mouse": 200,
    "clock": 150,
    "bowl": 20,
    "chair": 300,
    "keyboard": 220,
    "laptop": 230,
    "orange": 250,
}

# Billing data
bill = {}
frame_count = 0
max_missing_frames = 60
# Object detection loop in a separate thread
def detect_objects():
    global frame_count
    cap = cv2.VideoCapture(0)  # Webcam
    while True:
        success, frame = cap.read()
        if not success:
            continue
        frame_count += 1

        results = model.track(frame, conf=0.2, persist=True, device=device)
        detected_counts = {}
        for result in results:
            if result.boxes.cls is not None:
                for cls_val in result.boxes.cls.tolist():
                    cls_idx = int(cls_val)
                    label = model.names[cls_idx]
                    detected_counts[label] = detected_counts.get(label, 0) + 1

        for label, count in detected_counts.items():
            unit_price = price_list.get(label, 0)
            bill[label] = {
                'label': label,
                'quantity': count,
                'price': unit_price,
                'last_seen': frame_count
            }

        # Remove outdated items
        labels_to_remove = []
        for label, data in bill.items():
            if label not in detected_counts:
                if frame_count - data.get('last_seen', frame_count) > max_missing_frames:
                    labels_to_remove.append(label)
        for label in labels_to_remove:
            del bill[label]

# Start detection thread
threading.Thread(target=detect_objects, daemon=True).start()

# API route to return bill as JSON
@app.route("/api/bill")
def get_bill():
    # Add total price
    total = sum(item['quantity'] * item['price'] for item in bill.values())
    response = {
        'items': list(bill.values()),
        'total': total
    }
    return jsonify(response)

# Run server on local network
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
