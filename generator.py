"""
This script loads `insec_cams.tsv` and download available videos into `IPCam` directory.
"""

import os
import re
import cv2
import time
import datetime
import itertools

import ephem
import pandas as pd
import numpy as np
import multiprocessing as mp
from cargan.utils.IPCam import IPCam

import sys
from jinja2 import Environment, FileSystemLoader

DATASET = 'insecam_traffic.tsv'
DEFAULT_DIR = '/media/dat/dataset1/IPCam'
DEFAULT_SEQ = 40
CPU = 4
CURRENT_LIGHTING = 0.4


def _main_():
    df = pd.read_csv(DATASET, sep='\t')
    df['current_lighting'] = df.T.apply(lambda row: has_day_light(row))
    _ = df.pop('Notes')
    df = df.sort_values(by=['current_lighting'], axis=0, ascending=False)
    df = df.reset_index(drop=True)
    df = df[df.current_lighting >= CURRENT_LIGHTING]
    print("Number of IP Cameras: %s" % len(df))

    pool = mp.Pool(CPU)
    chunks = np.array_split(df, indices_or_sections=int(df.shape[0]/CPU))
    results = pool.map(func, chunks)
    results = [item for sublist in results for item in sublist]


def func(chunk):
    results = []
    for index, ip in chunk.iterrows():
        camera = IPCam(ip['ip_cam'])
        parent_dir = '_'.join([re.sub(r'\W+', '', ip['code']),
                               re.sub(r'\W+', '', ip['city']),
                               str(ip['zip']).strip('-_')])

        local_timezone = calculate_local_time(re.sub(r'[+-]\W+', '', ip['timezone']))
        sequence_dir = '_'.join(['day' if ip['current_lighting'] > 0.1 else 'night',
                                 datetime.datetime.now(local_timezone).strftime("%Y%m%d_%H%M%S")])

        output_dir = os.path.join(DEFAULT_DIR, parent_dir, sequence_dir)
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as error:
                print(error)
                pass
        with open(os.path.join(output_dir, 'ip_address.txt'), 'w') as fio:
            fio.write(ip['ip_cam'])
        try:
            start_time = time.time()
            frames = camera.get_sequence(DEFAULT_SEQ)
            [cv2.imwrite(os.path.join(output_dir, '%s.jpg' % frame_id), frame)
             for frame_id, frame in enumerate(frames)]
            print("--- Completed %s in  %s seconds ---" % (os.path.join(parent_dir, sequence_dir),
                                                           time.time() - start_time))
            results.append({parent_dir: {
                ip['ip_cam']: [os.path.join(parent_dir, sequence_dir, '%s.jpg' % frame_id)
                               for frame_id in range(len(frames))]
            }})
        except Exception as error:
            print("Remove %s because of %s" % (output_dir, error))
            pass

    return results


def has_day_light(row):
    o = ephem.Observer()
    o.lat = str(row['latitude'])
    o.long = str(row['longitude'])
    sun = ephem.Sun(o)
    return float(sun.alt)


def calculate_local_time(timezone):
    hours, minutes = timezone.split(":")
    offset_in_seconds = 60*(60*int(hours) + int(minutes))
    tzinfo = CustomTimezone(offset_in_seconds)
    return tzinfo


class CustomTimezone(datetime.tzinfo):

    def __init__(self, offset_in_seconds, name=None):
        self.offset = datetime.timedelta(seconds=offset_in_seconds)
        self.name = name or self.__class__.__name__

    def utcoffset(self, dt):
        return self.offset

    def tzname(self, dt):
        return self.name

    def dst(self, dt):
        return datetime.timedelta(0)

if __name__ == '__main__':
    _main_()
