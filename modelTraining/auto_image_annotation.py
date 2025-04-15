import os

import numpy as np
import yaml
from PIL import Image
from pathlib import Path

from yolox.yolo import init_yolo, read_frame

parent_dir = Path.cwd().parent

with open(parent_dir / 'config.yaml', 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    yolo = config['yolo']

image_folder_path = "image"
label_folder_path = "label"

model = init_yolo(parent_dir / yolo['model'])

image_files = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]


def image_loader(files):
    for file in files:
        yield Image.open(file)  # Yield one image at a time

image_iterator = image_loader(image_files)

while True:
    try:
        img = next(image_iterator)  # Get next image
        frame = np.array(img)
        results = read_frame(model, frame)

        h, w, _ = frame.shape  # height, width

        base_name = os.path.splitext(os.path.basename(img.filename))[0]  # Removes the file extension
        output_file =  os.path.join(label_folder_path, base_name + '.txt')

        with open(output_file, "w") as f:

            boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            class_ids = results.boxes.cls.cpu().numpy()  # Class IDs

            for class_id, box in zip(class_ids, boxes_xyxy):
                x_min, y_min, x_max, y_max = box

                center_x = ((x_min + x_max) / 2) / w  # Normalize X center
                center_y = ((y_min + y_max) / 2) / h  # Normalize Y center
                box_width = (x_max - x_min) / w  # Normalize width
                box_height = (y_max - y_min) / h  # Normalize height

                f.write(f"{int(class_id)} {center_x:.4f} {center_y:.4f} {box_width:.4f} {box_height:.4f}\n")

        img.close()  # Close after processing to free memory
    except StopIteration:
        print("All images processed.")
        break  # Exit loop when all images are processed
