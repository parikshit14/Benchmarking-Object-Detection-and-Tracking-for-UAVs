# Inference detection with multi-object tracking algorithms

Make sure environment is setup according to the detectors, in general trackers do not require major dependency installations.

### Tracking setups for BoTSORT

This setup is only required for botsort and rest of the trackers can be used without any setup

Step 1: Setup
```bash
cd trackers/BoT-SORT

pip3 install -r requirements.txt

python3 setup.py develop

pip3 install cython_bbox

pip3 install faiss-cpu
```

## Run Inference
Supported trackers include:
1. ByteTrack
2. DeepSORT
3. StrongSORT
4. SMILEtrack
5. BotSORT

> **Note:** Please replace `<tracker_name>` with the tracker name as per requirement; e.g., *test_bytetrack.py*, *test_botsort.py*, *test_deepsort.py*, *test_smiletrack.py*, *test_strongsort.py*

---
### EdgeYOLO
Activate EdgeYOLO environment (if not already setup, see training readme file)
Considering the working directory to be edgeyolo

```bash
python tools/test_<tracker_name>.py --source ../../sample_video/input_video.mp4 --weights weight/best_DET_trained.pth --show
```

**Extra:** 
- For faster inference, add `--trt` flag with the command and use the `weight/trt_inferencing.pt` weights
- For Tracking with ReID (available in BotSORT and SMILEtrack) use `--with-reid` flag

---
### CEASC
Activate CEASC environment (if not already setup, see training readme file)
```bash
python tools/test_<tracker_name>.py output/dynamic_gfl_res18_visdrone.py ../../sample_video/input_video.mp4 output/epoch_15.pth --show
```
**Extra:** 
- For Tracking with ReID (available in BoTSORT and SMILEtrack) use `--with-reid` flag


---
### Yolov8
Activate Yolo environment (if not already setup, see training readme file)
Considering the working directory to be yolov8
```bash
python tools/test_<tracker_name>.py --source ../../sample_video/input_video.mp4 --model weights/visdrone_s.pt --show
```

**Extra:** 
- For faster inference, use the `weights/visdrone_s.engine` weights
- For Tracking with ReID (available in BotSORT and SMILEtrack) use `--with-reid` flag

Go back to [main documentation](../readme.md)



