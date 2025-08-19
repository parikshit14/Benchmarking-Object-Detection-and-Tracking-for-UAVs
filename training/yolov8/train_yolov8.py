from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(data="training/utils/visdrone.yaml", epochs=100, imgsz=640)
