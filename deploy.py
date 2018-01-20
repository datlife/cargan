from __future__ import print_function

import os
import tensorflow as tf
from utils.converter import load_graph_from_pb

# TF Libraries to export model into .pb file
from tensorflow.python.client import session
from tensorflow.python.saved_model import signature_constants
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.tools.graph_transforms import TransformGraph


def _main_():
    # #################
    # Setup export path
    ###################
    model_filename = './faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb'

    version    = 1
    model_name = 'faster_rcnn_inception'
    base_dir   = './detector'
    output_dir = os.path.join(base_dir, model_name)
    export_path = os.path.join(output_dir, str(version))

    # ######################
    #  Interference Pipeline
    # ######################
    input_names = 'image_tensor'
    output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']

    with tf.Session() as sess:
        input_tensor = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name=input_names)
        # ###################
        # load frozen graph
        # ###################
        graph_def = load_graph_from_pb(model_filename)
        outputs = tf.import_graph_def(graph_def,
                                      input_map={'image_tensor': input_tensor},
                                      return_elements=output_names,
                                      name='')
        outputs = [sess.graph.get_tensor_by_name(ops.name +':0')for ops in outputs]
        outputs = dict(zip(output_names, outputs))

    # #####################
    # Quantize Frozen Model
    # #####################
    transforms = ["add_default_attributes",
                  "quantize_weights", "round_weights",
                  "fold_batch_norms", "fold_old_batch_norms"]

    quantized_graph = TransformGraph(input_graph_def=graph_def,
                                     inputs=input_names,
                                     outputs=output_names,
                                     transforms=transforms)

    # #####################
    # Export to TF Serving#
    # #####################
    # Reference: https://github.com/tensorflow/models/tree/master/research/object_detection

    with tf.Graph().as_default():
        tf.import_graph_def(quantized_graph, name='')

        # Optimizing graph
        rewrite_options = rewriter_config_pb2.RewriterConfig(optimize_tensor_layout=True)
        rewrite_options.optimizers.append('pruning')
        rewrite_options.optimizers.append('constfold')
        rewrite_options.optimizers.append('layout')
        graph_options = tf.GraphOptions(rewrite_options=rewrite_options, infer_shapes=True)

        # Build model for TF Serving
        config = tf.ConfigProto(graph_options=graph_options)

        # @TODO: add XLA for higher performance (AOT for ARM, JIT for x86/GPUs)
        # https://www.tensorflow.org/performance/xla/
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with session.Session(config=config) as sess:
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            tensor_info_inputs = {'inputs': tf.saved_model.utils.build_tensor_info(input_tensor)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                            inputs     = tensor_info_inputs,
                            outputs    = tensor_info_outputs,
                            method_name= signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={'predict_images': detection_signature,
                                       signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: detection_signature,
                                       },
            )
            builder.save()

    print("\n\nModel is ready for TF Serving. (saved at {}/saved_model.pb)".format(export_path))


if __name__ == '__main__':
    _main_()
