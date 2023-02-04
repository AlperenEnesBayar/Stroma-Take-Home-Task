import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np


from models.experimental import attempt_load
from YOLO_utils.datasets import LoadStreams, LoadImages
from YOLO_utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from YOLO_utils.plots import plot_one_box, plot_one_box_track
from YOLO_utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from yolox.tracker.byte_tracker import BYTETracker


class BYTETrackerArgs:
    track_thresh: float = 0.5
    track_buffer: int = 15
    match_thresh: float = 0.8
    min_box_area: float = 10.0
    mot20: bool = False


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def detect():
    source = 'bnn_data/images/test'
    weights = 'trained_models/best.pt'
    device = '0'
    img_size = 640
    classes = 0, 1
    conf_thres = 0.55
    iou_thres = 0.45
    imgsz = img_size
    augment = False
    agnostic_nms = False

    # Directories
    # Initialize
    device = select_device(device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    model = TracedModel(model, device, img_size)
    model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    tracker = BYTETracker(BYTETrackerArgs)

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            trackInput = []
            detectInput = []
            pred_bbox = []

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    cls = int(cls)
                    conf = float(conf)
                    x1 = int(x1.cpu().detach().numpy())
                    x2 = int(x2.cpu().detach().numpy())
                    y1 = int(y1.cpu().detach().numpy())
                    y2 = int(y2.cpu().detach().numpy())

                    trackInput.append([x1, y1, x2, y2, conf])
                    detectInput.append([[x1, y1, x2, y2], None])

                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                trackInput = np.array(trackInput)
                try:
                    trackInput = torch.from_numpy(trackInput)
                    online_targets = tracker.update(trackInput, [img_size, img_size], (img_size, img_size))
                except:
                    empty_bbox = torch.empty(1, 6)
                    online_targets = tracker.update(empty_bbox, [img_size, img_size], (img_size, img_size))

                for i, t in enumerate(online_targets):
                    tlwh = t.tlwh
                    tid = t.track_id
                    tlbr = t.tlbr
                    pred_bbox.append([tlbr, tid, detectInput[i][-1]])

                    for s_track in detectInput:
                        iou = bb_intersection_over_union(s_track[0], tlbr)
                        print(iou)
                        if iou > 0.5:
                            x_coor = int(s_track[0][2] + 6)
                            y_coor = int(s_track[0][1])
                            cv2.putText(im0, "ID: " + str(tid), org=(x_coor, y_coor), fontFace=0, fontScale=0.5,
                                        color=[0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
                            cv2.putText(im0, "ID: " + str(tid), org=(x_coor, y_coor), fontFace=0, fontScale=0.5,
                                        color=[225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # if len(pred_bbox) != 0:
                #     for i in range(len(pred_bbox)):
                #         plot_one_box_track(im0, detectInput[i][0][2], detectInput[i][0][1], pred_bbox[i][1])

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            cv2.imshow("Test", im0)
            cv2.waitKey(1 )  # 1 millisecond

            # Save results (image with detections)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    with torch.no_grad():
        detect()
