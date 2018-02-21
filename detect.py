import os
import cv2
import numpy as np
import tensorflow as tf

from cargan.detector.Detector import Detector
from cargan.utils.parser import load_data, parse_label_map
from cargan.utils.ops import compute_nms

DATASET_PATH    = './IPCam'
DETECTORS       = ['faster_rcnn_inception_resnet_v2_atrous_coco', 'faster_rcnn_nas_coco']
COCO_LABEL      = './cargan/detector/label_maps/mscoco.pbtxt'
ALLOWED_OBJECTS = ['car', 'truck', 'bus']


def _main_():
    session = tf.Session()
    detectors = []
    for idx, model in enumerate(DETECTORS):
        detector = Detector(model_name=model,
                            model_path=os.path.abspath(os.path.join('./cargan', 'detector', model)),
                            label_map=parse_label_map(COCO_LABEL),
                            server='localhost:%s' % (9000 + idx),
                            allowed_gpu_fraction=0.00)
        detectors.append(detector)

    for city, info in load_data(DATASET_PATH).items():
        for timestamp, data in info.items():
            detected_vehicles = 0
            label_file = os.path.join(DATASET_PATH, city, timestamp, 'labels.csv')
            if not os.path.isfile(label_file):
                with open(label_file, 'w') as fio:
                    for img_path in data['images']:
                        image = cv2.imread(DATASET_PATH + img_path)
                        h, w, _ = image.shape
                        boxes, scores = predict_using_ensemble(detectors,
                                                               resize_keep_ratio(image, min_width=600),
                                                               session,
                                                               score_threshold=0.3)
                        if len(boxes):
                            scaled_boxes = [box * np.array([h, w, h, w]) for box in boxes]
                            detected_vehicles += len(scaled_boxes)
                            for bbox, score in zip(scaled_boxes, scores):
                                y1, x1, y2, x2 = bbox
                                fio.write('{} {:.6f} {} {} {} {}\n'.format(img_path.split('/')[-1], score,
                                                                           int(x1), int(y1), int(x2), int(y2)))

                if detected_vehicles < 2:
                    no_object_in_seq = os.path.join(DATASET_PATH, city, timestamp)
                    print("No objects appear in this sequence. Removing directory %s" % no_object_in_seq)
                    try:
                        os.rmdir(no_object_in_seq)
                    except Exception as e:
                        print(e)
                        pass
                print("Label is created in %s" % city+timestamp)

    # Turn off server
    [detector.stop() for detector in detectors]
    session.close()


def predict_using_ensemble(detectors, resized_image, session, score_threshold=0.3):
    bboxes = []
    scores = []
    for model in detectors:
        detections = model.predict(resized_image)
        detections = remove_objects_not_in(ALLOWED_OBJECTS,
                                           detections,
                                           threshold=score_threshold)
        if zip(*detections):
            print(detections)
            boxes, classes, confidences = zip(*detections)
            bboxes.append(boxes)
            scores.append(confidences)

    if bboxes:
        merged_result = compute_nms(np.concatenate(bboxes),
                                    np.concatenate(scores), session,
                                    iou_thresh=0.5)
        return merged_result['bboxes'], merged_result['scores']
    else:
        return [], []


def remove_objects_not_in(allowed_objects, detections, threshold):
    # Filter out results that is not reach a threshold
    filtered_detections = [(box, idx, score) for box, idx, score in zip(*detections)
                           if (score > threshold) and (idx in allowed_objects)]
    return filtered_detections


def resize_keep_ratio(image, min_width=600):
    # resize
    new_height = int(float(image.shape[1]) * float(min_width / float(image.shape[0])))
    resized_img = cv2.resize(image, (new_height, min_width), cv2.INTER_CUBIC)
    return resized_img


if __name__ == '__main__':
    _main_()