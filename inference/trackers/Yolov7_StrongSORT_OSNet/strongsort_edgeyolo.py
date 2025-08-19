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


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


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

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
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
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    cls_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    # source = str(source)
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    # is_file = Path(source).suffix[1:] in (VID_FORMATS)
    # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # if is_url and is_file:
    #     source = check_file(source)  # download

    # exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    # save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    # save_dir = Path(save_dir)
    # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device("")
    ####
    detector = Detector
    model = detector(
        weight_file="/home/parikshit/Desktop/edgeyolo/vid_trained_model/edgeyolo_visdrone_300epoch/best.pth",
        conf_thres=0.25,
        nms_thres=0.55,
        input_size=[640, 640],
        fuse=True,
        fp16=False,
        use_decoder=False
    )

    names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)


    tracker = StrongSORT(
                strong_sort_weights,
                device,
                fp16=False,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        
    tracker.model.warmup()
    outputs = [None]
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    ###
    video_path = "/home/parikshit/Downloads/drone_sample_videos/"
    video_name = "visdrone_126_0001.mp4"

    video_out_path = os.path.join('.', 'EdgeYOLO_Strongsort_'+video_name)

    cap = cv2.VideoCapture(video_path+video_name)
    ret, frame = cap.read()

    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))
    ###

    frame_count = 0
    total_fps = 0
    total_fps_tracker= 0
    total_combined_fps = 0
    # for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
    while ret:
        frame_count += 1
        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        xywh_bboxs = []
        confs = []
        oids = []
        # output = []
        # for result in results:
        #     if result is not None:
        #         for *xywh, obj, conf, cls in result:
        #             x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xywh)
        #             xywh_obj = [x_c, y_c, bbox_w, bbox_h ]
        #             xywh_bboxs.append(xywh_obj)
        #             confs.append(conf.item())
        #             oids.append(int(cls))
        if results[0] is not None:            
            for *xywh, obj, conf, cls in results[0]:
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xywh)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append(torch.Tensor([conf]))
                oids.append(int(cls))

            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
            print("##########",confss,"#########")
            # confs = torch.Tensor(confs)
            oids = torch.Tensor(oids)
            # pass detections to strongsort
            # xywh_bboxs = torch.Tensor(xywh_bboxs)
            start_time_tracker = time.time()
            outputs = tracker.update(xywhs, confss, oids, frame)
            end_time_tracker = time.time()
            fps_detection = 1 / (end_time - start_time)
            fps_tracker = 1/ (end_time_tracker - start_time_tracker)
            fps_combined = 1/ (end_time_tracker - start_time)
            print("detection fps - ",fps_detection)
            print("tracking fps - ",fps_tracker)
            print("combined fps - ",fps_combined)


            total_fps += fps_detection
            total_fps_tracker+=fps_tracker
            total_combined_fps += fps_combined
            if len(outputs) > 0:
                # print("@@in if@@@@@@@@@@@")
                bbox = outputs[:, :4]
                identities = outputs[:, 4]
                # print("identities:-------",identities)
                object_id = outputs[:, 5]

                # draw_boxes(frame, outputs, model.class_names)
                online_im = plot_tracking(frame,bbox,identities,object_id,cls_names,frame_id=frame_count, fps=total_combined_fps / frame_count)
            else:
                online_im = frame
        else:
            online_im = frame
        cap_out.write(online_im)
        # frame=cv2.resize(frame,(1280,720))
        cv2.imshow("test",online_im)
        # cv2.moveWindow("test",0,0)
        ret, frame = cap.read()

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #     # draw boxes for visualization
    #     if len(outputs) > 0:
    #         for j, (output, conf) in enumerate(zip(outputs, confs)):

    #             bboxes = output[0:4]
    #             id = output[4]
    #             cls = output[5]
    #             c = int(cls)  # integer class
    #             id = int(id)  # integer id
    #             label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
    #                 (f'{id}' if hide_class else f'{id} {names[c]}'))
    #             plot_one_box(bboxes, frame, label=label, color=colors[int(cls)], line_thickness=1)
    #         if show_vid:
    #             cv2.imshow("test", frame)
    #             cv2.waitKey(1)  # 1 millisecond
    #         cap_out.write(frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     ret, frame = cap.read()
    cap.release()
    # cap_out.release()
    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS detection: {avg_fps:.3f}")
    
    avg_fps_tracker = total_fps_tracker / frame_count
    print(f"Average FPS tracking: {avg_fps_tracker:.3f}")

    avg_fps_combined = total_combined_fps / frame_count
    print(f"Average FPS tracking: {avg_fps_combined:.3f}")

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
