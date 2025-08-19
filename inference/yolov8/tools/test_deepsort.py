import os
import random
from ultralytics import YOLO
import cv2
import argparse
from collections import deque
import numpy as np
import torch
import time
import sys

sys.path.append('../trackers/')
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

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
video_out_path = 'Yolov8s_DeepSORT_'+video_name
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

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, w, h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i])) + f'{clss[i]}'
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text + f'{cls_names[int(clss[i])]}', (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def fps_calculator(start_time,current_time):
    # if current_time - start_time >= 1:
    fps = 1 / (current_time - start_time)
    print(f"FPS: {fps:.2f}")


frame_count = 0
total_fps = 0
total_fps_tracker= 0
total_fps_detection = 0
cfg_deep = get_config()
cfg_deep.merge_from_file("../trackers/deep_sort_pytorch/configs/deep_sort.yaml")
deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
detection_threshold = 0.1
results_list = []
cls_names = ['group', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
if(torch.cuda.is_available()):
    device = 0
else:
    device = 'cpu'

print("Device being used is", device)

while ret:
    frame_count += 1
    start_time = time.time()
    results = model.predict(frame, device=0, classes = [1,2,3,4,5,6,7,8,9,10,11], verbose=False )
    end_time_detection = time.time()
    total_fps_detection += 1/(end_time_detection - start_time)
    xywh_bboxs = []
    confs = []
    oids = []
    for result in results:
        detections = []
        for bbox, cl, conf in zip(result.boxes.xywh.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            x, y, w, h = bbox
            score = conf
            class_id = cl
            class_id = int(class_id)
            if score > detection_threshold:
                xywh_bboxs.append([x, y, w, h])
                confs.append([score])
                oids.append(class_id)
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        # When no object is detected to avoid crashing in deepsort.update this if conition is added
        if torch.equal(xywhs, torch.tensor([])):
            continue
        
        tracking_input = detections
        img_size_passed = [frame.shape[1], frame.shape[0]]
        
        start_time_tracker = time.time()
        width, height = frame.shape[1], frame.shape[0]
        outputs = deepsort.update(xywhs, confss, oids, frame)
        end_time_tracker = time.time()
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_tracker = 1/ (end_time_tracker - start_time_tracker )
        
        total_fps += fps
        total_fps_tracker+=fps_tracker

        if len(outputs) > 0:
                bbox = outputs[:, :4]
                identities = outputs[:, -2]
                # print("identities:-------",identities)
                object_id = outputs[:, -1]

                # draw_boxes(frame, outputs, model.class_names)
                online_im = plot_tracking(frame,bbox,identities,object_id,cls_names,frame_id=frame_count, fps=total_fps / frame_count)
        else:
                online_im = frame
    
    cap_out.write(online_im)
    if args.show:
        cv2.imshow("DeepSORT tracking with Yolov8s", online_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

print("Average FPS: ", total_fps/frame_count)
print("Average FPS Tracker: ", total_fps_tracker/frame_count)
print("Average FPS Detection: ", total_fps_detection/frame_count)