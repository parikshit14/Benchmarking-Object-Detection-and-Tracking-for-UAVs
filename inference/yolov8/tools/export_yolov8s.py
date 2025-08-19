from ultralytics import YOLO

# # Load the YOLO11 model
# model = YOLO("weights/visdrone_s.pt")

# # Export the model to TensorRT format
# model.export(format="engine", half = True)  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("/home/parikshit/Desktop/BEL_codebase/inference/yolov8/weights/visdrone_s.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")