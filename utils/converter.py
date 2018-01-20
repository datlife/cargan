"""Utilities to freeze model for interference

"""
import sys
import tensorflow as tf

from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2


def load_graph_from_pb(model_filename):
    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(data)
            # sm = saved_model_pb2.SavedModel()
            # sm.ParseFromString(data)
            # if 1 != len(sm.meta_graphs):
            #     print('More than one graph found. Not sure which to write')
            #     sys.exit(1)
            # # g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

    return graph_def
