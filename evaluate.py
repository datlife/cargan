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

from utils.ops import compute_iou, merge_dict, compute_nms
from utils.metrics import compute_average_precision
from utils.visualizer import visualize_boxes_and_labels_on_image_array, encode_image_array_as_png_str


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

    ground_truths = {os.path.basename(xml_file).split('.')[0]: parse_pascal_voc_annotation(xml_file)
                     for xml_file in annotation_files}
    # Start detection evaluation
    for idx, detection_file in enumerate(detection_files):
        run_eval_on(model_name=os.path.basename(detection_file).split('_det')[0],
                    detections=parse_detection_file(detection_file),
                    ground_truths=ground_truths,
                    log_dir=args.logdir)

    # Test ensemble
    detections = reduce(merge_dict, [parse_detection_file(det) for det in detection_files])
    detections = {img_id: compute_nms(detections[img_id]['bboxes'],
                                      detections[img_id]['scores'],
                                      iou_thresh=0.5) for
                  img_id in detections}

    run_eval_on(model_name='ensemble_model',
                detections=detections,
                ground_truths=ground_truths,
                log_dir=args.logdir)

    # Visualize result
    img_file_pattern = os.path.join(os.path.abspath('./test_data/JPEGImages'), '*.jpg')
    img_files = sorted(glob(img_file_pattern), key=os.path.getctime)

    visualize_tfimage(img_files,
                      ground_truths, detections,
                      min_confidence=0.2,
                      name_tag='ensemble_model',
                      log_dir=args.logdir)


def run_eval_on(model_name, detections, ground_truths, log_dir):
    scores, labels, ap = eval_ap_recall_per_class(
        ground_truths,
        detections,
        iou_threshold=IOU_THRESH)

    summarize(scores,
              labels,
              log_dir,
              run_name="IOU={:.2f}::AP={:.2f}::{}".format(IOU_THRESH, ap, model_name),
              num_thresholds=NUM_THRESHOLDS)


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


def visualize_tfimage(image_paths,
                      ground_truths,
                      detections,
                      min_confidence,
                      name_tag, log_dir, global_step=1):
    # a hacky way to not create a complete label dict
    category_index = {1: {'id': '2', 'name': 'car'}}

    # Read all images
    for img in image_paths:
        image = np.array(Image.open(img))
        img_id = os.path.basename(img).split('.')[0]

        ground_truth_boxes = np.array(ground_truths[img_id]['bboxes'])
        visualize_boxes_and_labels_on_image_array(image,
                                                  boxes=ground_truth_boxes[..., [1, 0, 3, 2]],
                                                  classes=None,
                                                  scores=None,
                                                  use_normalized_coordinates=False,
                                                  category_index=category_index,
                                                  max_boxes_to_draw=None)

        if img_id in detections:
            bboxes = np.array(detections[img_id]['bboxes'])
            scores = np.array(detections[img_id]['scores'])
            classes = ['1'] * len(scores)
            visualize_boxes_and_labels_on_image_array(image,
                                                      boxes=bboxes[..., [1, 0, 3, 2]],
                                                      classes=classes,
                                                      scores=scores,
                                                      use_normalized_coordinates=False,
                                                      category_index=category_index,
                                                      agnostic_mode=True,
                                                      min_score_thresh=min_confidence)
        fixed_width = 640
        new_height = int(float(image.shape[1]) * float(fixed_width / float(image.shape[0])))
        image = Image.fromarray(image)
        image.thumbnail((new_height, fixed_width))

        img_summary_op = tf.Summary(value=[
            tf.Summary.Value(
                tag="detection_samples/%s" % name_tag,
                image=tf.Summary.Image(
                    encoded_image_string=encode_image_array_as_png_str(
                        image)))
        ])
        summary_writer = tf.summary.FileWriter(log_dir)
        summary_writer.add_summary(img_summary_op, global_step)
        summary_writer.close()


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

    print('Saved summary in %s\n' % event_dir)
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


if __name__ == '__main__':
    _main_()