"""Utilities to freeze model for interference

"""
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile

def load_graph_from_pb(model_filename):
    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(data)
    return graph_def
