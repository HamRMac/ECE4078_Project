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
                boxes_out = []
                img_out = deepcopy(src)
                boxes = pred.boxes
                for box in boxes:
                    box_cord = box.xywh[0]
                    box_label = box.cls
                    label_str = pred.names[int(box_label)]
                    boxes_out.append([label_str, np.asarray(box_cord)])
                    # Draw
                    xyxy = ops.xywh2xyxy(box_cord)
                    x1, y1, x2, y2 = map(int, [xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                    colour = self.class_colour.get(label_str, (200, 200, 200))
                    img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), colour, thickness=2)
                    img_out = cv2.putText(img_out, label_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
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

        # get bounding box and class label for target(s) detected
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                # bounding format in [x, y, width, height]
                box_cord = box.xywh[0]

                box_label = box.cls  # class label of the box

                bounding_boxes.append([prediction.names[int(box_label)], np.asarray(box_cord)])

        return bounding_boxes


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
