import cv2
import numpy as np
import tensorflow as tf
import threading
import requests
import time

# Path to your TensorFlow saved model
model_path = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model"
model = tf.saved_model.load(model_path)

# Define pricing for detected items
price_list = {
    "bottle": 100,
    "cup": 80,
}

# Global variables for bill state and frame counting
bill = {}
frame_count = 0
max_missing_frames = 60

# Mapping from COCO label IDs to your item names
coco_labels = {
    44: "bottle",
    47: "cup"
}

# Endpoint from your Vercel deployment (note: vercel.json should map routes properly)
VERCEL_API = "https://automated-billing-system.vercel.app/api/bill"

def detect_objects():
    global frame_count, bill
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to capture frame.")
            continue

        frame_count += 1

        # Prepare the frame as input tensor
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run object detection
        detections = model(input_tensor)
        detected_counts = {}

        num_detections = int(detections.pop('num_detections'))
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
        detection_scores = detections['detection_scores'][0].numpy()

        for i in range(num_detections):
            score = detection_scores[i]
            if score > 0.2:
                label_id = detection_classes[i]
                label = coco_labels.get(label_id)
                if label and label in price_list:
                    detected_counts[label] = detected_counts.get(label, 0) + 1

        # Update the bill with current detections
        for label, count in detected_counts.items():
            unit_price = price_list[label]
            bill[label] = {
                "label": label,
                "quantity": count,
                "price": unit_price,
                "last_seen": frame_count
            }

        # Remove items not detected in recent frames
        labels_to_remove = [label for label, data in bill.items()
                            if frame_count - data["last_seen"] > max_missing_frames]
        for label in labels_to_remove:
            del bill[label]

        # Debug: Print current bill state
        print("Current Bill:", bill)

        # Post updated bill data to Vercel
        try:
            payload = {
                "items": list(bill.values()),
                "total": sum(item["quantity"] * item["price"] for item in bill.values())
            }
            response = requests.post(VERCEL_API, json=payload, timeout=2)
            if response.status_code != 200:
                print("Error posting to Vercel: Status code", response.status_code)
        except Exception as e:
            print("Exception posting to Vercel:", e)

        # Brief sleep to reduce CPU usage
        time.sleep(0.1)
    
    cap.release()

def main():
    detection_thread = threading.Thread(target=detect_objects, daemon=True)
    detection_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program.")

if __name__ == "__main__":
    main()
