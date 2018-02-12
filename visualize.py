import flask
from flask_paginate import Pagination, get_page_args
from cargan.utils.parser import load_data, flatten_dict

app  = flask.Flask(__name__, static_url_path="", static_folder="IPCam")
data = flatten_dict(load_data("IPCam")).items()

# Pagination Cfg
# ---------------
PER_PAGE = 3


@app.route('/', methods=['GET', 'POST'])
def index():
    current_page, per_page, offset = get_page_args()
    pagination = Pagination(total=int(len(data) / PER_PAGE),
                            page=current_page,
                            per_page=PER_PAGE,
                            search=False,
                            link_size='md',
                            show_single_page=False,
                            css_framework='bootstrap4')

    cities = data[offset: offset + PER_PAGE]

    return flask.render_template('index.html',
                                 pagination=pagination,
                                 cities=cities)


if __name__ == '__main__':
    app.run(host='localhost',  port=5000, debug=True)
