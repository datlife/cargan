import numpy as np
import tensorflow as tf


def compute_nms(boxes, scores, session, iou_thresh):
    """
    """
    bboxes = boxes[..., [1, 0, 3, 2]]  # change to y1, x1, y2, x2
    kept_indices = tf.image.non_max_suppression(bboxes, scores, max_output_size=200, iou_threshold=iou_thresh)

    boxes = tf.gather(bboxes, kept_indices)
    scores = tf.gather(scores, kept_indices)

    boxes, scores = session.run([boxes, scores])

    return {
        'bboxes':  boxes[..., [1, 0, 3, 2]],  # change back to x1, y1, x2, y2
        'scores': scores
    }


def merge_dict(a, b, path=None):
    "Merge two nested dict into one"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            else:  # leaf data
                a[key] = np.concatenate([a[key], b[key]])
        else:
            a[key] = b[key]
    return a


def compute_iou(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) matrix

    Args:
      boxes1: - a np.array of N boxes [N, 4]
      boxes2: - a np.array of M boxes [M, 4]

    Returns:
      a IoU matrix [N, M]

    """
    areas1 = area(boxes1)
    areas2 = area(boxes2)
    intersections = intersection(boxes1, boxes2)

    unions = (np.expand_dims(areas1, 1) +
              np.expand_dims(areas2, 0) - intersections)

    return np.where(np.equal(intersections, 0.0),
                    np.zeros_like(intersections),
                    np.divide(intersections, unions))


def area(boxes):
    """Computes area of boxes.
    Args:
      boxes: a np.array holding N boxes [N, 4]

    Returns:
      a tensor with shape [N] representing box areas.
    """
    y_min, x_min, y_max, x_max = np.split(boxes, 4, axis=-1)
    return np.squeeze((y_max - y_min) * (x_max - x_min), [1])


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Args:
      boxes1:   a np.array holding N boxes [N, 4]
      boxes2:  a np.array holding M boxes [N, 4]
    Returns:
        a np.array shape [N, M] representing pairwise intersections
    """
    y_min1, x_min1, y_max1, x_max1 = np.split(boxes1, 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = np.split(boxes2, 4, axis=-1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))

    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))

    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)

    return intersect_heights * intersect_widths
