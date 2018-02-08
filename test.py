import os
import sys
import time
import cv2
import numpy as np
from glob import glob

from utils.painter import draw_boxes
from utils.parser import parse_label_map
from utils.tfserving import DetectionClient, DetectionServer

TEST_DATA  = './test_data/JPEGImages'
OUTPUT_DIR = './test_data/Main/'

DETECTORS = [
    'faster_rcnn_inception_resnet_v2_atrous_coco',
    'faster_rcnn_nas_coco',
    'ssd_inception_v2_coco',
    'rfcn_resnet101_coco'
]
LABEL_MAPS = {
    'coco': './detector/label_maps/mscoco.pbtxt',
    'kitti': './detector/label_maps/kitti.pbtxt',
}

# @TODO: fail gracefully kill -9 $(lsof -t -i:9000 -sTCP:LISTEN)


def main():
    server = 'localhost:9000'

    # Prepare images
    glob_pattern = os.path.join(os.path.abspath(TEST_DATA), '*.jpg')
    img_files = sorted(glob(glob_pattern), key=os.path.getctime)

    for model_name in DETECTORS:
        # Init Detection Server
        model_path = os.path.join(sys.path[0], 'detector', model_name)
        tf_serving_server = DetectionServer(model=model_name, model_path=model_path).start()
        # Wait for server to start
        time.sleep(5.0)
        if tf_serving_server.is_running():
            print("\n\nInitialized TF Serving at {} with model {}".format(server, model_name))

            label_dict = parse_label_map(LABEL_MAPS['coco'] if 'coco' in model_name else LABEL_MAPS['kitti'])
            object_detector = DetectionClient(server, model_name, label_dict, verbose=True)

            evaluate(img_files, object_detector,
                     output_dir='./%s/%s' % (OUTPUT_DIR, model_name),
                     threshold=0.001)

            print("\nWaiting for last predictions before turning off...")
            time.sleep(5.0)
            tf_serving_server.stop()


def evaluate(images, detector, output_dir, threshold=0.2, fixed_width=600):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    performance = {}
    for img_path in images:

        img_id = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        # resize
        new_height = int(float(img.shape[1]) * float(fixed_width / float(img.shape[0])))
        img = cv2.resize(img, (new_height, fixed_width), cv2.INTER_CUBIC)
        h, w, _ = img.shape
        print("Predicting img {} | shape {}".format(img_id, img.shape))
        boxes, classes, scores = detector.predict(img, img_dtype=np.uint8, timeout=30.0)
        # Filter out results that is not reach a threshold
        filtered_outputs = [(box, idx, score) for box, idx, score in zip(boxes, classes, scores)
                            if (score > threshold) and (idx in ['car'])]

        if zip(*filtered_outputs):
            boxes, classes, scores = zip(*filtered_outputs)
            print("Detected {} cars with confidences {}\n".format(len(scores), scores))
            abs_size_boxes = [box * np.array([h, w, h, w]) for box in boxes]
            img = draw_boxes(img, abs_size_boxes, classes, scores)
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
