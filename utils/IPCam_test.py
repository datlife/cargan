import unittest
from IPCam import IPCam


class IPCamTest(unittest.TestCase):

    def test_setup(self):
        example = 'http://138.26.105.144:80/mjpg/video.mjpg?COUNTER'
        cam_test = IPCam(example)

        frame = cam_test.get_frame()
        print(frame.shape)

    def test_wrong_ip(self):
        example = 'http://wrongip.com:80'
        cam_test = IPCam(example)

    def test_dead_ip(self):
        example = 'http://0.0.0.1:8080'
        cam_test = IPCam(example)

if __name__ == '__main__':
    unittest.main()
