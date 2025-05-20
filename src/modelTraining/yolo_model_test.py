from ultralytics import YOLO
if __name__ == "__main__":
    model_A = YOLO("yolov8m-football.pt")

    for name, model in {"A": model_A}.items():
        metrics = model.val(data="football.yaml",
                            imgsz=640, conf=0.25, iou=0.45,
                            save_json=True, save_txt=True,
                            split="test")        # or 'val' if that's your hold‑out split
        print(f"{name} metrics:", metrics)       # dict with mAP50‑95, precision, recall, etc.