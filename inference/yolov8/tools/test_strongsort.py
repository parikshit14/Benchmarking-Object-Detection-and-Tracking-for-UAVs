import argparse
import time
import os

import sys
import numpy as np
from pathlib import Path
import torch
# import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics import YOLO
import cv2

sys.path.append('../trackers/Yolov7_StrongSORT_OSNet/')
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO model video processing")
    parser.add_argument('--model', type=str, default='..', help='Path to YOLO model')
    parser.add_argument('--source', type=str, default='..', help='Path to input video')
    parser.add_argument('--show', action='store_true', help='Show video output')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=80, help='filter out tiny boxes')
    parser.add_argument('--strong_sort_weights', type=str, default='../trackers/Yolov7_StrongSORT_OSNet/weights/osnet_x0_25_msmt17.pt', help='path to strong sort')
    parser.add_argument('--config_strongsort', type=str, default='../trackers/Yolov7_StrongSORT_OSNet/strong_sort/configs/strong_sort.yaml', help='path to strong sort')
    parser.add_argument('--half', type=bool, default=False, help='use FP16 half-precision inference')
    parser.add_argument('--device',  default='cuda', help='device for inference')
    parser.add_argument('--hide_labels', type=bool, default=False, help='hide labels')
    parser.add_argument('--hide_conf', type=bool, default=False, help='hide confidences')
    parser.add_argument('--hide_class', type=bool, default=False, help='hide class')
    args = parser.parse_args()
    return args



def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, clss, cls_names, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
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

def main():
    args = parse_args()

    model = YOLO(args.model)
    video_name = os.path.basename(args.source)
    video_out_path = 'Yolov8s_StrongSORT_'+video_name
    print(f"Saving video to {video_out_path}")

    cap = cv2.VideoCapture(args.source)
    ret, frame = cap.read()
    width, height = frame.shape[1], frame.shape[0]
    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                            (frame.shape[1], frame.shape[0]))
    
    names = ['group', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
    cfg = get_config()
    cfg.merge_from_file(args.config_strongsort)

    tracker = StrongSORT(
                args.strong_sort_weights,
                args.device,
                fp16=False,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
    # tracker.model.warmup()

    frame_count = 0
    total_fps = 0
    total_fps_tracker= 0
    total_fps_detection = 0
    detection_threshold = 0.1

    while ret:
        frame_count += 1
        start_time = time.time()
        results = model.predict(frame, device=0, classes = [1,2,3,4,5,6,7,8,9,10,11], verbose=False)
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
            
            outputs = tracker.update(xywhs, confss, oids, frame)
            end_time_tracker = time.time()
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            fps_tracker = 1/ (end_time_tracker - start_time_tracker )
            
            total_fps += fps
            total_fps_tracker+=fps_tracker

            if len(outputs) > 0:
                    # print("@@in if@@@@@@@@@@@")
                    bbox = outputs[:, :4]
                    identities = outputs[:, 4]
                    # print("identities:-------",identities)
                    object_id = outputs[:, 5]

                    # draw_boxes(frame, outputs, model.class_names)
                    online_im = plot_tracking(frame,bbox,identities,object_id,names,frame_id=frame_count, fps=total_fps / frame_count)
            else:
                    online_im = frame
            
        
        cap_out.write(online_im)
        if args.show:
            cv2.imshow('StrongSORT tracking with Yolov8s', online_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        ret, frame = cap.read()

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()

    print("Average FPS: ", total_fps/frame_count)
    print("Average FPS Tracker: ", total_fps_tracker/frame_count)
    print("Average FPS Detection: ", total_fps_detection/frame_count)

if __name__ == "__main__":
    main()