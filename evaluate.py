import os
import sys
import time
import cv2
import numpy as np
from glob import glob

from utils.painter import draw_boxes
from utils.parser import parse_label_map
from utils.tfserving import DetectionClient, DetectionServer

TEST_DATA = './test_data/JPEGImages'
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
OUTPUT = './test_data/Main/'


def main():
    server = 'localhost:9000'

    # Prepare images
    glob_pattern = os.path.join(os.path.abspath(TEST_DATA), '*.jpg')
    img_files = sorted(glob(glob_pattern), key=os.path.getctime)

    for model_name in ['yolov2']:
        # Init Detection Server
        model_path = os.path.join(sys.path[0], 'detector', model_name)
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
            object_detector = DetectionClient(server, model_name, label_dict, verbose=False)

            # EVALUATE
            evaluate(img_files, object_detector,
                     output_dir='./%s/%s' % (OUTPUT, model_name),
                     threshold=0.3)

            # Stop server
            print("\nWaiting for last predictions before turning off...")
            time.sleep(5.0)
            tf_serving_server.stop()


def evaluate(images, detector, output_dir, threshold=0.2,):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    performance = {}
    for img_path in images:

        img_id = img_path.split('/')[-1]
        print("Processing img %s" % img_id)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        boxes, classes, scores = detector.predict(img, img_dtype=np.uint8, timeout=30.0)
        # Filter out results that is not reach a threshold
        filtered_outputs = [(box, idx, score) for box, idx, score in zip(boxes, classes, scores)
                            if (score > threshold) and (idx in ['car'])]

        if zip(*filtered_outputs):
            boxes, classes, scores = zip(*filtered_outputs)
            print("Detected {} cars with confidences {}\n".format(len(scores), scores))
            boxes = [box * np.array([h, w, h, w]) for box in boxes]
            img = draw_boxes(img, boxes, classes, scores)
            performance[img_id] = (boxes, classes, scores)
        else:
            print("No car was found.\n")

        output_path = os.path.join(output_dir, img_path.split('/')[-1])
        cv2.imwrite(output_path, img)

    with open(os.path.join(output_dir, '%s_det_val_car.txt' % detector.model), 'w') as f:
        for img_id in performance.keys():
            result = performance[img_id]
            for bbox, cls, score in zip(*result):
                y1, x1, y2, x2 = bbox
                f.write('{} {:.6f} {} {} {} {}\n'.format(img_id.split('.')[0], score, int(x1), int(y1), int(x2), int(y2)))
if __name__ == '__main__':
    main()
