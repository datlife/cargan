import time
import cv2
import multiprocessing as mp
from cargan.utils.IPCam import IPCam

IP_CAMS = ['http://138.26.105.144:80/mjpg/video.mjpg?COUNTER',
           'http://74.78.100.54:80/mjpg/video.mjpg?COUNTER',
           'http://220.254.136.170:80/cgi-bin/camera?resolution=640&amp;amp;quality=1',
           'http://174.6.126.86:80/mjpg/video.mjpg?COUNTER'
           'http://108.53.114.166:80/mjpg/video.mjpg?COUNTER'
           'http://138.16.170.10:80/mjpg/video.mjpg?COUNTER',
           'http://166.155.71.82:8080/mjpg/video.mjpg?COUNTER',
           'http://96.81.44.121:80/mjpg/video.mjpg?COUNTER',
           'http://166.165.35.32:80/mjpg/video.mjpg?COUNTER',
           'http://166.165.35.32:8888/mjpg/video.mjpg?COUNTER'
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
           'http://203.250.69.180:80/mjpg/video.mjpg?COUNTER',
           'http://110.5.60.225:8081/mjpg/video.mjpg?COUNTER',
           'http://195.208.255.162:80/mjpg/video.mjpg?COUNTER',
           'http://1.250.137.134:80/mjpg/video.mjpg?COUNTER',
           'http://195.189.181.205:80/mjpg/video.mjpg?COUNTER',
           'http://1.250.137.135:80/mjpg/video.mjpg?COUNTER',
           'http://94.214.183.239:8001/mjpg/video.mjpg?COUNTER',
           'http://96.67.45.105:80/mjpg/video.mjpg?COUNTER',
           'http://192.65.213.243:80/mjpg/video.mjpg?COUNTER',
           'http://89.184.13.11:83/webcapture.jpg?command=snap&amp;channel=1?1517889249',
           'http://89.184.13.11:81/webcapture.jpg?command=snap&amp;channel=1?1517889271',
           'http://202.142.10.11:80/SnapshotJPEG?Resolution=640x480&amp;amp;Quality=Clarity&amp;amp;1517889339',
           'http://192.121.142.39:80/mjpg/video.mjpg?COUNTER',
           'http://166.154.60.23:8080/mjpg/video.mjpg?COUNTER',
           'http://50.73.9.194:80/mjpg/video.mjpg?COUNTER',
           'http://106.244.199.94:8081/mjpg/video.mjpg?COUNTER',
           'http://80.13.228.129:8080/oneshotimage1?1518051386',
           'http://184.151.45.56:80/cgi-bin/camera?resolution=640&amp;amp;quality=1&amp;amp;Language=0&amp;amp;1518051410',
           'http://78.28.217.43:80/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER',
           'http://50.47.66.157:80/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER',
           'http://166.154.143.99:8080/mjpg/video.mjpg?COUNTER',
           'http://64.77.205.67:80/mjpg/video.mjpg?COUNTER',
           'http://173.10.83.218:80/mjpg/video.mjpg?COUNTER',
           'http://1.250.137.147:80/mjpg/video.mjpg?COUNTER',
           'http://61.60.112.229:80/mjpg/video.mjpg?COUNTER',
           'http://166.140.204.12:80/mjpg/video.mjpg?COUNTER',
           'http://193.180.141.127:80/mjpg/video.mjpg?COUNTER',
           'http://1.250.137.145:80/mjpg/video.mjpg?COUNTER',
           'http://1.250.137.148:80/mjpg/video.mjpg?COUNTER',
           'http://1.250.137.151:80/mjpg/video.mjpg?COUNTER',
           'http://166.140.204.12:80/mjpg/video.mjpg?COUNTER',
           'http://128.119.174.134:80/mjpg/video.mjpg?COUNTER',
           'http://1.250.137.165:80/mjpg/video.mjpg?COUNTER',
           'http://80.249.245.18:80/cgi-bin/viewer/video.jpg?r=1518053015'
           ]


def main():
    print(len(IP_CAMS))
    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    pool.map(func, enumerate(IP_CAMS))
    print("--- Completed in  %s seconds ---" % (time.time() - start_time))


def func(tuple_args):
    idx, ip = tuple_args
    camera = IPCam(ip)
    try:
        frame = camera.get_frame()
        print("Saving image {} shape {} on IP {}".format(idx, frame.shape, ip))
        cv2.imwrite('./test_data/%s.jpg' % idx, frame)
    except Exception as e:
        print(e, "ID: %s IP: %s" % (idx, ip))
        pass


if __name__ == '__main__':
    main()