import sys
import cv2
import json
from contextlib import closing
import numpy as np
if sys.version_info[0] < 3:
    import urllib2 as urltool
else:
    import urllib.request as urltool

import multiprocessing as mp

# @TODO: add test
# @TODO: skip if fail to get frame
class IPCam(object):
    """
    A definition of an IP camera
    """
    def __init__(self, ip):
        self.ip = ip

    def get_sequence(self, sequence_len, timeout=5.0):
        seq = []
        for i in range(sequence_len):
            seq.append(self.get_frame(timeout))
        return seq

    def get_frame(self, timeout):
        stream = urltool.urlopen(self.ip, timeout=timeout)

        if stream is None:
            return None

        bytes = b''
        while True:
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b + 2]
                bytes = bytes[b + 2:]
                img_arr = np.asarray(bytearray(jpg), dtype=np.uint8)

                frame = cv2.imdecode(img_arr, 1)
                return frame


