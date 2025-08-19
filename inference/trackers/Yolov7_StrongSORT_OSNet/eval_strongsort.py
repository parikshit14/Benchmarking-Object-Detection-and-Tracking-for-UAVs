import argparse
import time
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
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


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
import sys
sys.path.insert(0, "../")
from edgeyolo.detect import Detector, TRTDetector, draw

import os
import glob
from functools import partial

#tracker_ = BYTETracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=25)

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

def get_all_folders(folder_path):
    # def list_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))

def get_folder_names(paths):
  """
  Extracts folder names from a list of paths.

  Args:
      paths: A list of strings representing file paths.

  Returns:
      A list of strings containing the folder names from the input paths.
  """
  folder_names = []
  for path in paths:
        # Split the path by the separator ("/")
        parts = path.split("/")
        # Get the last element (folder name)
        folder_names.append(parts[-1])
  return folder_names

@torch.no_grad()
def run(source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=True,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference):
):
    save_dir = Path('EdgeYOLO_StrongSORT')
    save_dir.mkdir(parents=True,exist_ok=True)
    device = select_device(device)
    
    # Initialize
    # set_logging()
    
    stride = 1
    detector = Detector
    model = detector(
        weight_file="/home/parikshit/Desktop/edgeyolo_visdrone/best_DET_trained.pth",
        conf_thres=0.25,
        nms_thres=0.55,
        input_size=[640, 640],
        fuse=True,
        fp16=False,
        use_decoder=False
    )
    names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    

   
    t0 = time.time()
    
    folder_list = get_all_folders('/home/parikshit/Desktop/VisDroneVID/VisDrone2019-VID-test-dev/VisDrone2019-VID-test-dev/sequences')
    # args = parse_args()
    for folder in folder_list:
        # Create tracker
        # change the opt to args from main_botsort and run
        # initializing tracker inside coz, the net frame obj-id start after the prev frame eg: if seq1 has max objid as 356 then seq2 starts from 357 which should not be the case
        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file(opt.config_strongsort)


        tracker = StrongSORT(
                    strong_sort_weights,
                    device,
                    half,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
                )
        print("folder name",folder.split("/")[-1])
        dataset = LoadImages(folder, img_size=1920, stride=stride)
        

        results = []
        fn = 0
        for path, img, im0s, vid_cap in dataset:
            # print("img",img)
            # cv2.imshow("temp",im0s)
            # cv2.waitKey(0)
            # print("---------im0s------",im0s)
            fn += 1

            # timer.tic()

            # img = torch.from_numpy(img).to(device)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            # img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # if img.ndimension() == 3:
            #     img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = model(im0s)
            # print("---pred---",pred)
            # Apply NMS
            # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # t2 = time_synchronized()

            # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # else:
                # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # Run tracker
                xywh_bboxs = []
                confs = []
                oids = []
                # bbox_result = results[:4]
                #print(bbox_result)
                for result in pred:
                    if result is not None:
                        for *xywh, obj, conf, cls in result:
                            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xywh)
                            xywh_obj = [x_c, y_c, bbox_w, bbox_h ]
                            xywh_bboxs.append(xywh_obj)
                            confs.append(conf.item())
                            oids.append(int(cls))
                confs = torch.Tensor(confs)
                oids = torch.Tensor(oids)
                # pass detections to strongsort
                xywh_bboxs = torch.Tensor(xywh_bboxs)
                # start_time_tracker = time.time()
                online_targets = tracker.update(xywh_bboxs, confs, oids, im0s)
                # print(online_targets)
                # trackerTimer.toc()
                # timer.toc()

                if len(online_targets) > 0:
                    for j, (output, conf) in enumerate(zip(online_targets, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id}' if hide_class else f'{id} {names[c]}'))
                        plot_one_box(bboxes, im0s, label=label, color=colors[int(cls)], line_thickness=1)

                        # save results
                        results.append(
                            f"{fn},{id},{bboxes[0]:.2f},{bboxes[1]:.2f},{bboxes[2]:.2f},{bboxes[3]:.2f},{conf:.2f},{c+1},-1,-1\n"
                            # f"{fn},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},-1,-1,-1,-1\n"
                        )
                        # print("----results----",results)
                        # if True:  # Add bbox to image
                        #     # if opt.hide_labels_name:
                        #     #     label = f'{tid}, {int(tcls)}'
                        #     # else:
                        #     label = f'{tid}, {names[int(tcls)]}'
                        #     plot_one_box(tlbr, im0s, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)

                # Stream results
                if True:
                    cv2.imshow('StrongSORT', im0s)
                    cv2.waitKey(1)  # 1 millisecond

        res_file = 'EdgeYOLO_StrongSORT' + "/" + folder.split("/")[-1] + ".txt"
        with open(res_file, 'w') as f:
            f.writelines(results)
        print(f"save results to {res_file}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)