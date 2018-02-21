import re
import flask
import pandas as pd
from flask_paginate import Pagination, get_page_args
from cargan.utils.parser import load_data, flatten_dict

app  = flask.Flask(__name__, static_url_path="", static_folder="/media/dat/dataset/DETRAC/OUTPUT")

# Pagination Cfg
PER_PAGE = 6


@app.route('/', methods=['GET', 'POST'])
def index():
    data = flatten_dict(load_data("/media/dat/dataset/DETRAC/OUTPUT", load_gt=True), current_level=0, max_level=1)
    current_page, per_page, offset = get_page_args()
    pagination = Pagination(total=int(len(data)),
                            page=current_page,
                            per_page=PER_PAGE,
                            search=False,
                            link_size='md',
                            show_single_page=False,
                            css_framework='bootstrap4')

    cities = data.items()[offset: offset + PER_PAGE]
    return flask.render_template('index.html',
                                 pagination=pagination,
                                 cities=cities,
                                 pandas=pd,
                                 custom_sort=custom_list_sort)


def custom_list_sort(a_list, reverse=False):
    return sorted(a_list,
                  key=lambda img_file: int(re.findall('\d+', img_file.split('/')[-1].split('.')[0])[0]),
                  reverse=reverse)

if __name__ == '__main__':
    app.run(host='169.237.118.28',  port=8080, debug=True)
