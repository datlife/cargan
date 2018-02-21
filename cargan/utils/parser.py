import re
import os
import csv
from itertools import islice

import numpy as np


def parse_inputs(filename, label_dict):
    """Read input file and convert into inputs, labels for training

    Args:
      filename: text file - has content format as
          image_path, x1, x2, y1, y2, label1
          image_path, x1, x2, y1, y2, label2
      label_dict:  an encoding dictionary -
        mapping class names to indices

    Returns:
      inputs :  a list of all image paths to dataset
      labels :  a dictionary,
        key : image_path
        value: all objects in that image

    @TODO: when dataset is large, we should consider this method a generator
    """
    training_instances = dict()
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in islice(reader, 1, None):
            if not line:
                continue  # Ignore empty line

            img_path = line[0]
            cls_name = line[-1]
            x1, y1, x2, y2 = [float(x) for x in line[1:-1]]
            an_object = [y1, x1, y2, x2, label_dict[cls_name]]

            if img_path in training_instances:
                training_instances[img_path].append(an_object)
            else:
                training_instances[img_path] = [an_object]
    inputs = training_instances.keys()
    labels = {k: np.stack(v).flatten() for k, v in training_instances.items()}

    return inputs, labels


def parse_label_map(label_map_path):
    """Parse label map file into a dictionary
    Args:
      label_map_path:

    Returns:
      a dictionary : key: obj_id value: obj-name
    """
    # match any group having language of {id:[number] .. name:'name'}
    parser = re.compile(r'id:[^\d]*(?P<id>[0-9]+)\s+name:[^\']*\'(?P<name>[\w_-]+)\'')

    with open(label_map_path, 'r') as f:
        lines = f.read().splitlines()
        lines = ''.join(lines)

        # a tuple (id, name)
        result = parser.findall(lines)
        label_map_dict = {int(item[0]): item[1] for item in result}

        return label_map_dict


def load_data(starting_path, file_extensions=['jpg', 'png', 'jpeg'], load_gt=False):
    data = {'': {}}
    for dir_path, subdir_list, file_names in os.walk(starting_path):
        d = data
        dir_path = dir_path[len(starting_path):]
        for sub_dir in dir_path.split(os.sep):
            based = d
            d = d[sub_dir]
        if subdir_list:
            for dn in subdir_list:
                d[dn] = {}
        else:
            label_path = os.path.join(starting_path + dir_path, 'labels.csv')
            labels = None
            if os.path.isfile(label_path):
                labels = parse_detection_file(label_path) if not load_gt else parse_ground_truth_file(label_path)

            based[sub_dir] = {'images': [os.path.join(dir_path, f)
                                         for f in file_names if f.split('.')[-1] in file_extensions],
                              'detections':  labels if labels else None}

    return data['']


def parse_ground_truth_file(gt_file):
    ground_truths = {}
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines][1:]

        for line in splitlines:
            img_id = line[0]
            obj_bbox = np.array(line[1:5], dtype=np.float)
            obj_idx = float(line[-1])
            if img_id not in ground_truths.keys():
                ground_truths[img_id] = {
                    'scores': [obj_idx],
                    'bboxes': [obj_bbox]
                }
            else:
                ground_truths[img_id]['scores'].append(obj_idx)
                ground_truths[img_id]['bboxes'].append(obj_bbox)
    return ground_truths


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


def flatten_dict(d, current_level, max_level):
    def expand(key, value, curr_level, max):
        if isinstance(value, dict) and current_level < max_level:
            return [(key + '::' + k, v) for k, v in flatten_dict(value, curr_level+1, max).items()]
        else:
            return [(key, value)]
    items = [item for k, v in d.items() for item in expand(k, v, curr_level=current_level, max=max_level)]
    return dict(items)
