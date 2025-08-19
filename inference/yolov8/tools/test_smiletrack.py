# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import numpy as np
import torch
import cv2
import sys

from ultralytics import YOLO
from tracker.tracking_utils.timer import Timer

import os
import glob
from functools import partial
sys.path.append('../trackers/SMILEtrack/SMILEtrack_Official')
from tracker.mc_SMILEtrack import SMILEtrack
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO model video processing")
    parser.add_argument('--model', type=str, default='..', help='Path to YOLO model')
    parser.add_argument('--source', type=str, default='..', help='Path to input video')
    parser.add_argument('--show', action='store_true', help='Show video output')
    parser.add_argument("--ablation", dest="ablation", default=False, action="store_true", help="ablation ")
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=20, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")
    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")
    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"../trackers/weights/smiletrack_weights/ver12.pt",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')
    args = parser.parse_args()
    
    return args

args = parse_args()

model = YOLO(args.model)
video_name = os.path.basename(args.source)
video_out_path = 'Yolov8s_SMILETrack_'+video_name
print(f"Saving video to {video_out_path}")

cap = cv2.VideoCapture(args.source)
ret, frame = cap.read()
width, height = frame.shape[1], frame.shape[0]
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, clss, cls_names, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i])) + f'{clss[i]}'
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text + f'{cls_names[int(clss[i])]}', (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

args_track = parse_args()
tracker = SMILEtrack(args_track)


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
    results = model.predict(frame, device=0, verbose=False, classes = [1,2,3,4,5,6,7,8,9,10,11] )
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
        online_targets = tracker.update(tracking_input, frame)
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
        cv2.imshow('SMILEtrack tracking with Yolov8s', online_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

print("Average FPS: ", total_fps/frame_count)
print("Average FPS Tracker: ", total_fps_tracker/frame_count)
print("Average FPS Detection: ", total_fps_detection/frame_count)