# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
from functools import partial

import sys
from pathlib import Path
import torch.backends.cudnn as cudnn
from numpy import random


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse
import os
import os.path as osp
import time
import warnings
import mmcv
import numpy as np
import mmcv
import torch
from mmcv import DictAction, Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import cv2
#from yolox.tracking_utils.timer import Timer

#from yolox.tracker.byte_tracker import BYTETracker
import sys

sys.path.append('../trackers/Yolov7_StrongSORT_OSNet/')
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         setup_multi_processes, update_data_root)
from torch.utils.data import Dataset, DataLoader

# from yolov7.models.experimental import attempt_load
# from yolov7.utils.datasets import LoadImages, LoadStreams
# from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
#                                   check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
# from yolov7.utils.torch_utils import select_device, time_synchronized
# from yolov7.utils.plots import plot_one_box
#from yolox.utils.visualize import plot_tracking


def plot_tracking(image, tlwhs, obj_ids, clss, cls_names, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

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

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def xyxy_to_xywh(xyxy):
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



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('video_path', help ='path of the video')
    parser.add_argument('checkpoint', help='checkpoint file')
    
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    
    parser.add_argument('--threshold', default=0.3, type=float,
                help='detection threshold for bounding box visualization'
                )

    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
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


def convert_polygon(
        result,
        score_thr=0.3,

):
    from matplotlib.patches import Polygon

    ms_bbox_result, ms_segm_result = result, None
    if isinstance(ms_bbox_result, dict):
        result = (ms_bbox_result['ensemble'],
                  ms_segm_result['ensemble'])

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        

    num_bboxes = 0
    ret_label = None
    ret_bbox = None
    #ret_polygon = None
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        ret_bbox = bboxes
        ret_label = labels[:num_bboxes]

    return {'labels': ret_label,
            'bboxes': ret_bbox,
            }


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    #update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)


    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    
    cfg.device = get_device()
    cls_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']


    # build the model and load checkpoint
    #cfg.model.train_cfg = None

    model = init_detector(cfg, args.checkpoint)
    #model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cuda')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    #if 'CLASSES' in checkpoint.get('meta', {}):

    model.CLASSES = checkpoint['meta']['CLASSES']
    

    cap = mmcv.VideoReader(args.video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_name = "processed_" + args.video_path.split('/')[-1]
    print(f"Saving video to {save_name}")

    width = int(cap.width) if isinstance(cap.width, np.ndarray) else cap.width
    height = int(cap.height) if isinstance(cap.height, np.ndarray) else cap.height

    out = cv2.VideoWriter(
        save_name, fourcc, cap.fps,
        (width, height)
    )

    frame_count = 0 
    total_fps = 0
    total_fps_tracker= 0
    total_comb_fps =0

    names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    #initialize StrongSORT
    cfg_track = get_config()
    cfg_track.merge_from_file(args.config_strongsort)

    tracker = StrongSORT(
                args.strong_sort_weights,
                args.device,
                args.half,
                max_dist=cfg_track.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg_track.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg_track.STRONGSORT.MAX_AGE,
                n_init=cfg_track.STRONGSORT.N_INIT,
                nn_budget=cfg_track.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg_track.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg_track.STRONGSORT.EMA_ALPHA,
            )
        
    # tracker.model.warmup()
    outputs = [None]
    

    for frame in mmcv.track_iter_progress(cap):
        # Increment frame count.
        frame_count += 1
        start_time = time.time()
        result = inference_detector(model, frame)
        end_time = time.time() 

        bbox_result = result[0]
        #print(bbox_result)

        if bbox_result is not None:
            results = convert_polygon(result)

            bboxes = []
            scores = []
            oids = []  
            for i, cls in zip(results['bboxes'], results["labels"]):
                bboxes.append(xyxy_to_xywh(i[:4]))
                scores.append([i[4]])
                oids.append(int(cls))
            
            # print("bboxes:-------",bboxes)
            # print("scores:-------",scores)
            # print("oids:-------",oids)
            xywhs = torch.Tensor(bboxes)
            confss = torch.Tensor(scores)
            if torch.equal(xywhs, torch.tensor([])):
                continue
            #tracking_input = np.array([[*bbox, score, result] for bbox, score, result in zip(bboxes, scores, results["labels"])],  dtype=np.float32)
            #img_size_passed = [frame.shape[0], frame.shape[1]]

            start_time_tracker = time.time()

            
            outputs = tracker.update(xywhs, confss, oids, frame)
            end_time_tracker = time.time()
            fps = 1 / (end_time - start_time)
            fps_tracker = 1/ (end_time_tracker - start_time_tracker )
            
            total_fps += fps
            total_fps_tracker+=fps_tracker

            comb_fps = 1/ (end_time_tracker-start_time)
            total_comb_fps += comb_fps

            

            if len(outputs) > 0:
                bbox = outputs[:, :4]
                # convert bbox format from xywh to xyxy
                identities = outputs[:, 4]
                # print("identities:-------",identities)
                object_id = outputs[:, 5]

                # draw_boxes(frame, outputs, model.class_names)
                online_im = plot_tracking(frame,bbox,identities,object_id,cls_names,frame_id=frame_count, fps=total_comb_fps / frame_count)
            else:
                # print("outputs:-------",outputs)
                
                # print("in skip")
                online_im = frame

                # Saving Track predictions into a text file
                #with open(tracking_info_path, 'a') as f:
                #    line = f'{frame_idx},{track.track_id},{int(bbox[0])},{int(bbox[1])},{int(bbox[2] - bbox[0])},{int(bbox[3] - bbox[1])},-1,-1,-1\n'
                #    f.write(line)

            #online_targets = tracker.update(tracking_input, [height, width], img_size_passed)
            #end_time_tracker = time.time()


            

            #show_result = model.show_result(frame, result, score_thr=args.threshold)
            
            #show_result = frame.copy()
            #online_tlwhs = []
            #online_ids =[]
            #online_scores = []

            #for t in online_targets:
            #    tlwh = t.tlwh 
            #    track_id = t.track_id

            #    vertical = tlwh[2]/tlwh[3] > args.aspect_ratio_thresh

            #    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            #        online_tlwhs.append(tlwh)
            #        online_ids.append(track_id)
            #        online_scores.append(t.score)
            #        results_list.append( f"{frame_count}, {track_id}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f}, {t.score:.2f}, -1, -1, -1\n")

            #timer.toc()


            #online_im = plot_tracking(
            #        frame, online_tlwhs, online_ids, frame_id=frame_count, fps=1. / timer.average_time
            #    )
        else:

            #timer.toc()
            online_im = frame

        out.write(online_im)
        if args.show:
            cv2.imshow('StrongSORT tracking with CEASC', online_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    
    avg_fps_tracker = total_fps_tracker / frame_count
    print(f"Average FPS: {avg_fps_tracker:.3f}")

    avg_comb_fps = total_comb_fps/frame_count
    print(f"Average Combined FPS: {avg_comb_fps:.3f}")



if __name__ == '__main__':
    main()
