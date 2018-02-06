"""
Evaluate Detection Model's performance

Assumptions:
  * Annotation files follow PASCAL VOC format
  * Detection files follow PASCAL VOC Detection Evaluation format
"""
import os
import argparse
from PIL import Image
import numpy as np
from glob import glob
import tensorflow as tf
from functools import reduce
import xml.etree.ElementTree as ET
from tensorboard.plugins.pr_curve import summary

from utils.ops import compute_iou
from utils.metrics import compute_average_precision
from utils.visualizer import visualize_boxes_and_labels_on_image_array



parser = argparse.ArgumentParser()

parser.add_argument('--annotation_dir', default='./test_data/Annotations')

parser.add_argument('--detection_dir', default='./test_data/Main')

parser.add_argument('--logdir', default='/tmp/cargan',
                    help='Directory stores Tensorboard log')

IOU_THRESH = 0.5
NUM_THRESHOLDS = 100


def _main_():
    args = parser.parse_args()
    annotation__file_pattern = os.path.join(os.path.abspath(args.annotation_dir), '*.xml')
    detection_file_pattern   = os.path.join(os.path.abspath(args.detection_dir), '*.txt')

    annotation_files = sorted(glob(annotation__file_pattern), key=os.path.getctime)
    detection_files  = sorted(glob(detection_file_pattern), key=os.path.getctime)

    # Generate a dictionary for ground truths, whereas:
    #    key: filename without extension
    #    values: a list of dict
    ground_truths = {os.path.basename(xml_file).split('.')[0]: parse_pascal_voc_annotation(xml_file)
                     for xml_file in annotation_files}

    # Start detection evaluation
    combined = []
    for idx, detection_file in enumerate(detection_files):
        detections = parse_detection_file(detection_file)
        model_name = os.path.basename(detection_file).split('_det')[0]

        scores, labels, ap = eval_ap_recall_per_class(
            ground_truths,
            detections,
            iou_threshold=IOU_THRESH)
        run_name = "IOU={:.2f}::AP={:.2f}::{}".format(IOU_THRESH, ap, model_name)
        summarize(scores, labels, args.logdir, run_name, num_thresholds=NUM_THRESHOLDS)
        combined.append(detections)

    # Test ensemble
    model_name = 'ensemble_model'
    detections = reduce(merge, combined)
    for img_id in detections:
        detections[img_id] = run_nms(detections[img_id], iou_thresh=0.5)
    scores, labels, ap = eval_ap_recall_per_class(
        ground_truths,
        detections,
        iou_threshold=IOU_THRESH)

    run_name = "IOU={:.2f}::AP={:.2f}::{}".format(IOU_THRESH, ap, model_name)
    summarize(scores, labels, args.logdir, run_name, num_thresholds=NUM_THRESHOLDS)

    img_file_pattern = os.path.join(os.path.abspath('./test_data/JPEGImages'), '*.jpg')
    img_files = sorted(glob(img_file_pattern), key=os.path.getctime)
    for img in img_files:
        image = np.array(Image.open(img))
        image_id = os.path.basename(img).split('.')[0]

        bboxes = np.array(detections[img_id]['bboxes'])
        scores = np.array(detections[img_id]['scores'])
        classes = ['1'] * len(scores)

        visualize_boxes_and_labels_on_image_array(image,
                                                  boxes=bboxes,
                                                  classes=classes,
                                                  scores=scores,
                                                  category_index={1: {'id': '2', 'name': 'car'}},
                                                  min_score_thresh=0.3)


def run_nms(detections, iou_thresh):
    """

    :param detections: a dict {'bboxes':np.array, 'scores'}
    :param iou_thresh:
    :return:
    """
    bboxes = detections['bboxes']
    scores = detections['scores']

    bboxes = bboxes[..., [1, 0, 3, 2]]  # change to y1, x1, y2, x2
    kept_indices = tf.image.non_max_suppression(bboxes, scores, max_output_size=200, iou_threshold=iou_thresh)

    boxes = tf.gather(bboxes, kept_indices)
    scores = tf.gather(scores, kept_indices)

    with tf.Session() as sess:
        boxes, scores = sess.run([boxes, scores])

    return {
        'bboxes':  boxes[..., [1, 0, 3, 2]],  # change back to x1, y1, x2, y2
        'scores': scores
    }


def eval_ap_recall_per_class(ground_truths, detections, iou_threshold=0.5):

    """Generate Average Precision - Recall.

    First we compute the iou matrix between ground truths and detections. Its shape is
    `[num_ground_truths, num_detections]`.

    Reference:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00051000000000000000

    # True Positives:  Pr(detected  | object presents)
    # False Negatives: Pr(undetected| object presents) = 1 - Pr(detected | object presents)
    # False Positives: Pr(detected  | object not presents)
    """
    labels = []
    scores = []
    num_gt = 0
    for img_id in ground_truths.keys():
        gt_bboxes_per_img   = np.array(ground_truths[img_id]['bboxes'])
        num_gt += len(gt_bboxes_per_img)

        # If there is detected objects
        if img_id in detections.keys():
            pred_bboxes_per_img = np.array(detections[img_id]['bboxes'])
            iou_matrix = compute_iou(gt_bboxes_per_img, pred_bboxes_per_img)

            label = np.zeros(len(pred_bboxes_per_img), dtype=bool)
            detected = [False] * iou_matrix.shape[0]
            # Determine whether the detection is a True/False positive
            for col in range(iou_matrix.shape[1]):
                good_detections = iou_matrix[..., col] >= iou_threshold
                idx = int(np.argmax(good_detections))
                if np.count_nonzero(good_detections) > 0.0 and detected[idx] is False:
                    label[col] = 1.0     # mark as True Positive
                    detected[idx] = True
            scores.append(np.array(detections[img_id]['scores'], dtype=np.float32))
            labels.append(label)
            # # For debugging
            # if img_id == '19':
            #     np.set_printoptions(precision=2)
            #     print(iou_matrix)
            #     print(np.max(iou_matrix, axis=0))
            #     print("TP {} || FN {} || FP {}".format(np.count_nonzero(label),
            #                                            len(gt_bboxes_per_img) - np.count_nonzero(label),
            #                                            np.count_nonzero(~label)))

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    # sort scores in to generate PR curve
    sorted_indices = np.argsort(-scores)
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    true_positives  = labels
    false_positives = (1.0 - true_positives)
    recall    = np.cumsum(true_positives) / num_gt
    precision = np.cumsum(true_positives) / (np.cumsum(true_positives) + np.cumsum(false_positives))
    ap        = compute_average_precision(precision, recall)

    return scores, labels, ap


def summarize(scores, labels, logdir, run_name, num_thresholds):
    summary.op(
            tag='%s' % run_name.split('::')[-1],
            labels=tf.cast(labels, tf.bool),
            predictions=tf.cast(scores, tf.float32),
            num_thresholds=num_thresholds,
            display_name='PR Curve',
    )
    summary.op(
            tag='.summary_by_iou/%s' % run_name.split('::')[0].split('=')[-1],
            labels=tf.cast(labels, tf.bool),
            predictions=tf.cast(scores, tf.float32),
            num_thresholds=num_thresholds,
            display_name='PR Curve',
    )
    merged_op = tf.summary.merge_all()
    event_dir = os.path.join(logdir, run_name)
    writer = tf.summary.FileWriter(event_dir)
    with tf.Session() as session:
        writer.add_summary(session.run(merged_op), global_step=1)

    print('Save summary in %s\n' % event_dir)
    tf.reset_default_graph()
    writer.close()


def parse_pascal_voc_annotation(xml_file):
    """Parse annotation XML file in PASCAL format into a dictionary

    """
    tree = ET.parse(xml_file)
    objects = {
        'classes': [],
        'bboxes' : [],
    }
    for object in tree.findall('object'):
        bbox = object.find('bndbox')
        bbox = [int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text)]
        objects['classes'].append(object.find('name').text)
        objects['bboxes'].append(bbox)

    return objects


def parse_detection_file(detection_file):

    detections = {}
    with open(detection_file, 'r') as f:
        lines      = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]

        for line in splitlines:
            img_id    = line[0]
            obj_score = float(line[1])
            obj_bbox  = np.array(line[2:], dtype=np.float)

            if img_id not in detections.keys():
                detections[img_id] = {
                    'scores': [obj_score],
                    'bboxes': [obj_bbox]
                }
            else:
                detections[img_id]['scores'].append(obj_score)
                detections[img_id]['bboxes'].append(obj_bbox)

    return detections


def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:  # leaf data
                a[key] = np.concatenate([a[key], b[key]])
        else:
            a[key] = b[key]
    return a

if __name__ == '__main__':
    _main_()