import cv2
import time
import urllib2
import numpy as np
import multiprocessing as mp

IP_CAMS = ['http://138.26.105.144:80/mjpg/video.mjpg?COUNTER',
           'http://166.155.71.82:8080/mjpg/video.mjpg?COUNTER',
           'http://96.81.44.121:80/mjpg/video.mjpg?COUNTER',
           'http://166.165.35.32:80/mjpg/video.mjpg?COUNTER',
           'http://220.254.136.170/cgi-bin/camera?',
           'http://220.254.136.178/cgi-bin/camera?',
           'http://166.165.58.225:80/mjpg/video.mjpg?COUNTER',
           'http://64.77.205.67:80/mjpg/video.mjpg?COUNTER',
           'http://208.84.209.70/oneshotimage1?',
           'http://1.250.137.136:80/mjpg/video.mjpg?COUNTER',
           'http://96.91.10.218:80/mjpg/video.mjpg?COUNTER',
           'http://68.107.172.105:8082/oneshotimage1?1517441352',
           'http://211.23.31.192:81/GetData.cgi?CH=1',
           'http://192.82.150.11:8083/mjpg/video.mjpg?COUNTER',
           'http://216.67.98.12:80/mjpg/video.mjpg?COUNTER',
           'http://109.206.96.58:8080/cam_1.cgi',
           'http://110.5.60.225:8081/mjpg/video.mjpg?COUNTER'
           ]


def main():
    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    pool.map(func, enumerate(IP_CAMS))
    print("--- Completed in  %s seconds ---" % (time.time() - start_time))


def func(tuple_args):
    idx, ip_cam = tuple_args
    frame = cv2.imdecode(request_frame(ip_cam), 1)
    print("Saving image {} shape {} on IP {}".format(idx, frame.shape, ip_cam))
    cv2.imwrite('./test_data/%s.jpg' % idx, frame)


def request_frame(url):
    stream = urllib2.urlopen(url)
    bytes  = ''
    while True:
        bytes += stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a != -1 and b != -1:
            jpg   = bytes[a:b+2]
            bytes = bytes[b+2:]
            img_arr = np.asarray(bytearray(jpg), dtype=np.uint8)
            return img_arr


if __name__ == '__main__':
    main()