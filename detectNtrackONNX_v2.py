import onnx
import onnxruntime as ort
import numpy as np
import cv2

from YOLO_utils.plots import plot_one_box
from glob import glob


def draw_detections(image, boxes, scores, class_ids, class_names, mask_alpha=0.3):
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def rescale_boxes(boxes, input_shape):
    # Rescale boxes to original image dimensions
    input_shape = np.array([input_shape[3], input_shape[2], input_shape[3], input_shape[2]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([640, 640, 640, 640])
    return boxes


def extract_boxes(predictions, input_shape):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, input_shape)

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    return boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


ort_sess = ort.InferenceSession('trained_models/best.onnx', providers=['CUDAExecutionProvider'])
model_inputs = ort_sess.get_inputs()
input_shape = model_inputs[0].shape

names = ['bolt', 'nut']
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

conf_threshold = 0.5
iou_threshold = 0.5

for img_p in glob('bnn_data/images/train/*'):
    img_org = cv2.imread(img_p)
    img = np.moveaxis(img_org.copy(), 2, 0)
    img = np.expand_dims(img, 0)

    img = img.astype(np.float32)
    outputs = ort_sess.run(None, {'images': img})

    predictions = np.squeeze(outputs[0])

    obj_conf = predictions[:, 4]
    predictions = predictions[obj_conf > conf_threshold]
    obj_conf = obj_conf[obj_conf > conf_threshold]

    # Multiply class confidence with bounding box confidence
    predictions[:, 5:] *= obj_conf[:, np.newaxis]

    # Get the scores
    scores = np.max(predictions[:, 5:], axis=1)

    # Filter out the objects with a low score
    predictions = predictions[scores > conf_threshold]
    scores = scores[scores > conf_threshold]

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 5:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions, input_shape)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)

    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]

    combined_img = draw_detections(img_org, boxes, scores,
                                   class_ids, class_names=names, mask_alpha=0.4)

    cv2.imshow("Detected Objects", combined_img)
    cv2.waitKey(1)
