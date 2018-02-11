import cv2
from PIL.Image import Image
import numpy as np

import os
import sys
from cargan.utils.parser import explore, parse_label_map
from cargan.detector.Detector import Detector

DETECTORS  = ['ssd_inception_v2_coco',
              'faster_rcnn_nas_coco']
COCO_LABEL = './cargan/detector/label_maps/mscoco.pbtxt'

DEFAULT_DATA = './IPCam'


def _main_():
    faster_rcnn = Detector(model_name=DETECTORS[0],
                           model_path=os.path.abspath(os.path.join('./cargan', 'detector', DETECTORS[0])),
                           label_map=parse_label_map(COCO_LABEL))
    results = explore("./IPCam")
    for city, info in results.items():
        for timestamp, images in info.items():
            with open(os.path.join(DEFAULT_DATA, city, timestamp, 'labels.csv'), 'w') as fio:
                for img_path in images:
                    image = cv2.imread(os.path.join(DEFAULT_DATA, img_path))
                    h, w, _ = image.shape
                    boxes, scores, classes = faster_rcnn.predict(resize_keep_ratio(image, min_width=600))
                    scaled_boxes = [box * np.array([h, w, h, w]) for box in boxes]
                    for bbox, cls, score in zip(scaled_boxes, scores, classes):
                        y1, x1, y2, x2 = bbox
                        fio.write('{} {:.6f} {} {} {} {}\n'.format(img_path.split('/')[-1], score,
                                                                   int(x1), int(y1), int(x2), int(y2)))

            print("Label is created in %s" % city+timestamp)


def resize_keep_ratio(image, min_width=600):
    # resize
    new_height = int(float(image.shape[1]) * float(min_width / float(image.shape[0])))
    resized_img = cv2.resize(image, (new_height, min_width), cv2.INTER_CUBIC)
    return resized_img

if __name__ == '__main__':
    _main_()