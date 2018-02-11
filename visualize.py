import os
import sys
from glob import glob
from flask import Flask, render_template
from jinja2 import Environment, FileSystemLoader

test = []


def _main_():
    results = explore("./IPCam")
    for city, info in results.items():
        print(city)

    env = Environment(loader=FileSystemLoader(os.path.join(sys.path[0], 'templates')))
    template = env.get_template('overview.html')
    parsed_template = template.render(cities=results)

    with open(os.path.join('./IPCam', 'visualization.html'), "w") as fio:
        fio.write(parsed_template)


def explore(starting_path, file_extensions=['jpg', 'png', 'jpeg']):
    alld = {'': {}}

    for dirpath, dirnames, filenames in os.walk(starting_path):
        d = alld
        dirpath = dirpath[len(starting_path):]
        for subd in dirpath.split(os.sep):
            based = d
            d = d[subd]
        if dirnames:
            for dn in dirnames:
                d[dn] = {}
        else:
            based[subd] = ['.'+os.path.join(dirpath, f) for f in filenames if f.split('.')[-1] in file_extensions]

    return alld['']


if __name__ == '__main__':
    _main_()