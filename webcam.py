"""Demo YOLOv2 using webcam or any video source

In order to find a video source, type `ls -l /dev/video*` in UNIX terminal

"""
import os
import sys
import cv2
import time
import argparse
import yaml
import numpy as np
import multiprocessing as mp

from utils.parser import parse_label_map
from utils.painter import draw_boxes, draw_fps
from utils.tfserving import DetectionClient, DetectionServer
from utils.webcam import WebcamVideoStream


def main():
    model_name = 'ssd'
    model_path = os.path.join(sys.path[0], 'detector', model_name)

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--video_source', type=int, default=1,
                        help="Video device[default = 1]")

    # ############
    # Parse Config
    # ############
    ARGS = parser.parse_args()
    with open('config.yml', 'r') as stream:
        config = yaml.load(stream)

    interference = config['interference']
    label_dict = parse_label_map(config['label_map'])
    server     = interference['server']
    print(label_dict)
    # #####################
    # Init Detection Server
    # #####################
    tf_serving_server = DetectionServer(model=model_name, model_path=model_path)
    tf_serving_server.start()

    # Wait for server to start
    time.sleep(2.0)
    if tf_serving_server.is_running():
        print("Initialized TF Serving at {} with model {}".format(server, model_name))

        # ######################
        # Init Client & Streamer
        # ######################
        object_detector = DetectionClient(server, model_name, label_dict, verbose=True)
        video_capture   = WebcamVideoStream(ARGS.video_source,
                                            width=interference['frame_width'],
                                            height=interference['frame_height']).start()

        # ##########
        # Start Demo
        # ###########
        viewer = WebCamViewer(video_capture, object_detector, score_threshold=interference['score_threshold'])
        viewer.run()

        # ############
        # Stop server
        # ############
        print("\nWaiting for last predictions before turning off...")
        time.sleep(5.0)
        tf_serving_server.stop()


class WebCamViewer(object):
    def __init__(self, video_capture, detector, score_threshold=0.6):
        self.video    = video_capture
        self.detector = detector
        self.threshold = score_threshold

        self.input_q  = mp.Queue(maxsize=2)
        self.output_q = mp.Queue(maxsize=1)

        pool = mp.Pool(processes=1, initializer=self.worker)

    def run(self):
        boxes = []
        classes = []
        scores = []

        num_frames = 0
        detection_fps = 0
        detection_frames = 0

        start = time.time()
        while True:
            frame = self.video.read()

            if self.input_q.full():
                self.input_q.get()
            self.input_q.put(frame)

            if not self.output_q.empty():
                detection_frames += 1
                detection_fps = detection_frames / (time.time() - start)
                boxes, classes, scores = self.output_q.get()

            num_frames += 1
            camera_fps = num_frames / (time.time() - start)

            frame = draw_fps(frame, camera_fps, detection_fps)
            frame = draw_boxes(frame, boxes, classes, scores)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elapsed = (time.time() - start)
        print('[INFO] elapsed time (total): {:.2f}'.format(elapsed))
        print('[INFO] approx. FPS: {:.2f}'.format(num_frames / elapsed))
        print('[INFO] approx. detection FPS: {:.2f}'.format(detection_frames / elapsed))
        self.video.stop()
        cv2.destroyAllWindows()

    def worker(self):
        while True:
            frame = self.input_q.get()
            self.output_q.put(self.detect_objects_in(frame))

    def detect_objects_in(self, frame):
        h, w, _ = frame.shape
        boxes, classes, scores = self.detector.predict(frame, img_dtype=np.uint8)
        # Filter out results that is not reach a threshold
        filtered_outputs = [(box, idx, score) for box, idx, score in zip(boxes, classes, scores)
                            if score > self.threshold]
        if zip(*filtered_outputs):
            boxes, classes, scores = zip(*filtered_outputs)
            boxes = [box * np.array([h, w, h, w]) for box in boxes]
        else:  # no detection
            boxes, classes, scores = [], [], []
        return boxes, classes, scores


if __name__ == "__main__":
    main()
