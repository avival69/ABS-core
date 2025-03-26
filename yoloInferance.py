import cv2
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLOv8 model on GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8x").to(device)

# Price list keyed by object label
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

# Billing data structure
bill = {}

# Parameters
max_missing_frames = 24  # Remove an item if not detected for this many frames

# Initialize GUI
root = tk.Tk()
root.title("Automated Billing System")

# Set up the billing table
columns = ("Item", "Quantity", "Unit Price", "Total")
tree = ttk.Treeview(root, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col)
tree.pack(fill="both", expand=True)

# Label for total bill
total_label = tk.Label(root, text="Total: ₹0", font=("Arial", 16))
total_label.pack()

# Label to display video feed
video_label = tk.Label(root)
video_label.pack()

# Open webcam
cap = cv2.VideoCapture(0)
frame_count = 0  # Frame counter

def update_gui():
    """Update the billing table and total based on the current bill dictionary."""
    tree.delete(*tree.get_children())  # Clear existing table entries
    total_bill = 0
    for label, data in bill.items():
        quantity = data["quantity"]
        unit_price = data["price"]
        total_item = quantity * unit_price
        total_bill += total_item
        tree.insert("", "end", values=(label, quantity, unit_price, total_item))
    total_label.config(text=f"Total: ₹{total_bill}")

def process_frame():
    """Process a video frame: detect objects, update billing info, and refresh the GUI and video display."""
    global frame_count
    success, frame = cap.read()
    if not success:
        root.after(10, process_frame)
        return

    frame_count += 1

    # Perform tracking/detection
    results = model.track(frame, conf=0.2, persist=True)

    # Count detections by label for the current frame
    detected_counts = {}
    available_items = set(price_list.keys())

    for result in results:
        if result.boxes.cls is not None:
            for cls_val in result.boxes.cls.tolist():
                cls_idx = int(cls_val)
                label = model.names[cls_idx]
                if label in available_items:
                    detected_counts[label] = detected_counts.get(label, 0) + 1

    # Update the bill for each detected label
    for label, count in detected_counts.items():
        unit_price = price_list[label]
        bill[label] = {
            "label": label,
            "quantity": count,
            "price": unit_price,
            "last_seen": frame_count,
        }

    # Remove labels that have not been detected for a while
    labels_to_remove = [label for label, data in bill.items() if frame_count - data["last_seen"] > max_missing_frames]
    for label in labels_to_remove:
        del bill[label]

    # Update the billing GUI
    update_gui()

    # Annotate the frame with detections and display it
    annotated_frame = results[0].plot()
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, process_frame)

# Start the frame processing loop and GUI mainloop
process_frame()
root.mainloop()
