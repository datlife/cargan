import time
import numpy as np
from cargan.utils.tfserving import DetectionClient, DetectionServer


class Detector(object):

    def __init__(self, model_name, model_path, label_map, server='localhost:9000', verbose=True, allowed_gpu_fraction=0.0):
        self.host, self.port = server.split(':')
        self.server  = DetectionServer(model=model_name,
                                       model_path=model_path,
                                       port=int(self.port),
                                       per_process_gpu_memory_fraction=allowed_gpu_fraction).start()

        self.client  = DetectionClient(server=server,
                                       model=model_name,
                                       label_dict=label_map,
                                       verbose=verbose)
        time.sleep(2.0)

    def predict(self, resized_image, img_dtype=np.uint8, timeout=30.0):
        if self.server.is_running():
            boxes, classes, scores = self.client.predict(resized_image,
                                                         img_dtype=img_dtype,
                                                         timeout=timeout)

            return boxes, classes, scores

        return [], [], []

    def stop(self):
        self.server.stop()
