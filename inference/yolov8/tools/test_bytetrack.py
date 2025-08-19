import cv2
from ultralytics import YOLO
import argparse
import time
import torch
import numpy as np
import sys
import os
sys.path.append('../trackers/')
from ByteTrack.yolox.tracking_utils.timer import Timer
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.utils.visualize import plot_tracking


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO model video processing")
    parser.add_argument('--model', type=str, default='..', help='Path to YOLO model')
    parser.add_argument('--source', type=str, default='..', help='Path to input video')
    parser.add_argument('--show', action='store_true', help='Show video output')
    args = parser.parse_args()
    return args

args = parse_args()

model = YOLO(args.model)
video_name = os.path.basename(args.source)
video_out_path = 'Yolov8s_BYTEtrack_'+video_name
print(f"Saving video to {video_out_path}")

cap = cv2.VideoCapture(args.source)
ret, frame = cap.read()
width, height = frame.shape[1], frame.shape[0]
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))


args_track = argparse.Namespace(track_thresh=0.25, track_buffer=20, mot20=False, min_box_area=100,  new_track_thresh=0.6, match_thresh=0.9)
tracker = BYTETracker(args_track)

cls_names = ['group', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
detection_threshold = 0.1

frame_count = 0
total_fps = 0
total_fps_tracker= 0
total_fps_detection = 0
timer=Timer()
results_list = []
while ret:
    frame_count += 1
    start_time = time.time()
    results = model.predict(frame, device=0, classes = [1,2,3,4,5,6,7,8,9,10,11], verbose=False )
    end_time_detection = time.time()
    total_fps_detection += 1/(end_time_detection - start_time)
    for result in results:
        detections = []
        for bbox, cl, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = bbox
            score = conf
            class_id = cl
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score, class_id])

        detections = np.array(detections)
        if(detections.size == 0):
            online_im = frame
            break
        
        tracking_input = detections
        img_size_passed = [frame.shape[1], frame.shape[0]]
        
        start_time_tracker = time.time()
        width, height = frame.shape[1], frame.shape[0]
        online_targets = tracker.update(tracking_input, [width, height], img_size_passed)
        end_time_tracker = time.time()
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_tracker = 1/ (end_time_tracker - start_time_tracker )
        
        total_fps += fps
        total_fps_tracker+=fps_tracker

        show_result = frame.copy()
        online_tlwhs = []
        online_ids =[]
        online_scores = []
        online_cls = []

        for t in online_targets:
            tlwh = t.tlwh 
            track_id = t.track_id
        
            online_tlwhs.append(tlwh)
            online_ids.append(track_id)
            online_scores.append(t.score)
            online_cls.append(t.cls)
        

        timer.toc()
        online_im = plot_tracking(
                frame, online_tlwhs, online_ids, online_cls, cls_names, frame_id=frame_count, fps=total_fps/frame_count
            )
    
    cap_out.write(online_im)
    if args.show:
        cv2.imshow("BYTETrack Tracking with Yolov8s", online_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

print("Average FPS: ", total_fps/frame_count)
print("Average FPS Tracker: ", total_fps_tracker/frame_count)
print("Average FPS Detection: ", total_fps_detection/frame_count)