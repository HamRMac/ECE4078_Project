import cv2
import os
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops


class Detector:
    def __init__(self,
                 model_path: str = "",
                 imgsz: int = 320
                 ):
        self.model = YOLO(model_path)
        self.imgsz = int(imgsz)
        self.max_batch = 8
        # IoU threshold for suppressing overlapping boxes: keep highest-confidence
        self.overlap_iou_thresh = 0.5

        self.class_colour = {
            'orange': (0, 165, 255),
            'lemon': (0, 255, 255),
            'lime': (0, 255, 0),
            'tomato': (0, 0, 255),
            'capsicum': (255, 0, 0),
            'potato': (255, 255, 0),
            'pumpkin': (255, 165, 0),
            'garlic': (255, 0, 255)
        }

    def detect_single_image(self, img):
        """
        function:
            detect target(s) in an image
        input:
            img: image, e.g., image read by the cv2.imread() function
        output:
            bboxes: list of lists, box info [label,[x,y,width,height]] for all detected targets in image
            img_out: image with bounding boxes and class labels drawn on
        """
        bboxes = self._get_bounding_boxes(img)

        img_out = deepcopy(img)

        # draw bounding boxes on the image
        for bbox in bboxes:
            #  translate bounding box info back to the format of [x1,y1,x2,y2]
            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            # draw bounding box
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), self.class_colour[bbox[0]], thickness=2)

            # draw class label
            img_out = cv2.putText(img_out, bbox[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  self.class_colour[bbox[0]], 2)

        return bboxes, img_out

    def detect_batch(self, imgs: list) -> list:
        """Detect targets in a batch of images.

        Inputs:
          - imgs: list of BGR images (np.ndarray HxWx3)
        Outputs:
          - results: list of tuples (bboxes, img_out) per input image, where
              bboxes: [[label, np.array([x,y,w,h])], ...]
              img_out: visualised image with boxes/labels drawn
        Processes in micro-batches up to self.max_batch for efficiency.
        """
        if not imgs:
            return []
        results_all = []
        # Process in chunks
        for i in range(0, len(imgs), self.max_batch):
            chunk = imgs[i:i + self.max_batch]
            # Ultralytics can take a list of images directly
            predictions = self.model.predict(chunk, imgsz=self.imgsz, verbose=False)
            # predictions is a list aligned with chunk
            for pred, src in zip(predictions, chunk):
                img_out = deepcopy(src)
                boxes = pred.boxes
                # Collect boxes with confidence for suppression
                items = []  # each: {label: str, xywh: np.ndarray(4,), conf: float}
                for box in boxes:
                    box_cord = np.asarray(box.xywh[0])
                    box_label = int(box.cls)
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                    label_str = pred.names[box_label]
                    items.append({
                        'label': label_str,
                        'xywh': box_cord.astype(float),
                        'conf': conf
                    })

                # Suppress overlapping boxes (keep highest-confidence)
                kept = self._suppress_overlaps(items, self.overlap_iou_thresh)

                # Prepare outputs and draw
                boxes_out = []
                for it in kept:
                    boxes_out.append([it['label'], it['xywh']])
                    xyxy = ops.xywh2xyxy(it['xywh'])
                    x1, y1, x2, y2 = map(int, [xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                    colour = self.class_colour.get(it['label'], (200, 200, 200))
                    img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), colour, thickness=2)
                    img_out = cv2.putText(img_out, it['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

                results_all.append((boxes_out, img_out))
        return results_all

    def _get_bounding_boxes(self, cv_img):
        """
        function:
            get bounding box and class label of target(s) in an image as detected by YOLOv8
        input:
            cv_img    : image, e.g., image read by the cv2.imread() function
            model_path: str, e.g., 'yolov8n.pt', trained YOLOv8 model
        output:
            bounding_boxes: list of lists, box info [label,[x,y,width,height]] for all detected targets in image
        """

        # predict target type and bounding box with your trained YOLO

        predictions = self.model.predict(cv_img, imgsz=self.imgsz, verbose=False)

        # Collect boxes with confidence for suppression
        items = []  # each: {label: str, xywh: np.ndarray(4,), conf: float}
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                box_cord = np.asarray(box.xywh[0])
                box_label = int(box.cls)
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                items.append({
                    'label': prediction.names[box_label],
                    'xywh': box_cord.astype(float),
                    'conf': conf
                })

        # Suppress overlapping boxes (keep highest-confidence)
        kept = self._suppress_overlaps(items, self.overlap_iou_thresh)

        # Return in original format: [label, [x,y,w,h]]
        bounding_boxes = [[it['label'], it['xywh']] for it in kept]
        return bounding_boxes

    def _suppress_overlaps(self, items, iou_thresh=0.5):
        """Greedy suppression across all classes by IoU, keeping highest-confidence.

        items: list of dicts with keys: 'label' (str), 'xywh' (np.ndarray[4]), 'conf' (float)
        Returns a filtered list of items.
        """
        if not items:
            return []

        # Sort by confidence descending
        sorted_items = sorted(items, key=lambda d: d['conf'], reverse=True)
        kept = []
        for cand in sorted_items:
            keep = True
            for k in kept:
                if self._iou_xywh(cand['xywh'], k['xywh']) > iou_thresh:
                    keep = False
                    break
            if keep:
                kept.append(cand)
        return kept

    @staticmethod
    def _iou_xywh(a, b):
        """Compute IoU between two boxes in xywh (center x,y,width,height) format."""
        # Convert to xyxy
        ax, ay, aw, ah = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx, by, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        ax1, ay1 = ax - aw / 2.0, ay - ah / 2.0
        ax2, ay2 = ax + aw / 2.0, ay + ah / 2.0
        bx1, by1 = bx - bw / 2.0, by - bh / 2.0
        bx2, by2 = bx + bw / 2.0, by + bh / 2.0

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union


# FOR TESTING ONLY
if __name__ == '__main__':
    # get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    yolo = Detector(f'{script_dir}/model/yolov8_model.pt')

    img = cv2.imread(f'{script_dir}/test/test_image_1.png')

    bboxes, img_out = yolo.detect_single_image(img)

    print(bboxes)
    print(len(bboxes))

    cv2.imshow('yolo detect', img_out)
    cv2.waitKey(0)
