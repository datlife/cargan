import os
import sys
import time
import cv2
import numpy as np
from glob import glob

from utils.painter import draw_boxes
from utils.parser import parse_label_map
from utils.tfserving import DetectionClient, DetectionServer

TEST_DATA = './test_data'
DETECTORS = [
    'faster_rcnn_resnet101_kitti',
    'faster_rcnn_inception_resnet_v2_atrous_coco',
    'ssd_inception_v2_coco',
    'rfcn_resnet101_coco'
]

LABEL_MAPS = {
    'coco': './detector/label_maps/mscoco.pbtxt',
    'kitti': './detector/label_maps/kitti.pbtxt',
}


def main():
    server = 'localhost:9000'

    # Prepare images
    glob_pattern = os.path.join(os.path.abspath(TEST_DATA), '*')
    img_files = sorted(glob(glob_pattern), key=os.path.getctime)

    for model_name in ['rfcn_resnet101_coco']:
        model_path = os.path.join(sys.path[0], 'detector', model_name)

        # Init Detection Server
        tf_serving_server = DetectionServer(model=model_name, model_path=model_path).start()
        # Wait for server to start
        time.sleep(5.0)
        if tf_serving_server.is_running():
            print("\n\nInitialized TF Serving at {} with model {}".format(server, model_name))
            # Look up proper label map file
            if 'coco' in model_name:
                label_dict = parse_label_map(LABEL_MAPS['coco'])
            else:
                label_dict = parse_label_map(LABEL_MAPS['kitti'])

            # Init Detection Client
            object_detector = DetectionClient(server, model_name, label_dict, verbose=True)

            # EVALUATE
            visualize(img_files, object_detector, threshold=0.3)

            # Stop server
            print("\nWaiting for last predictions before turning off...")
            time.sleep(5.0)
            tf_serving_server.stop()


def visualize(images, detector, threshold=0.2):
    viz = './outputs'
    for img_path in images:
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        try:
            boxes, classes, scores = detector.predict(img, img_dtype=np.uint8, timeout=30.0)
        except Exception as e:
            print(e)
            time.sleep(1.0)
            continue

        # Filter out results that is not reach a threshold
        filtered_outputs = [(box, idx, score) for box, idx, score in zip(boxes, classes, scores)
                            if (score > threshold) and (idx in ['car', 'truck'])]

        if zip(*filtered_outputs):
            boxes, classes, scores = zip(*filtered_outputs)
            boxes = [box * np.array([h, w, h, w]) for box in boxes]
        img = draw_boxes(img, boxes, classes, scores)

        output_path = os.path.join(viz, img_path.split('/')[-1])
        cv2.imwrite(output_path, img)

if __name__ == '__main__':
    main()
