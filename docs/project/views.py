from flask import render_template
from flask.ext.assets import Environment, Bundle
from app import app, pages
import os
import os.path


NOTEBOOK_DIR = os.path.dirname(os.path.realpath(__file__)) + "/static/notebooks"
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/apiref/')
def apiref():
    return render_template('apiref.html')

@app.route('/tutorials/')
def tutorials():
    return render_template('tutorials.html')

@app.route('/showroom/')
def showroom():
    notebooks = get_notebooks()
    return render_template('showroom.html', examples=notebooks)

# assets
assets = Environment(app)

scss = Bundle('stylesheets/main.scss', filters='pyscss', output='gen/scss.css')
all_css = Bundle('vendor/*.css', scss, filters='cssmin', output="gen/all.css")
assets.register('css_all', all_css)

js = Bundle(
    'vendor/jquery-3.1.1.min.js',
    'vendor/jquery.timeago.js',
    'vendor/bootstrap.min.js',
    'vendor/showdown.min.js',
    'javascripts/*.js',
    filters='jsmin', output='gen/packed.js'
)
assets.register('js_all', js)

# utils
def get_abstract(fname):
    import json
    import os
    import markdown

    try:
        js = json.load(file(fname))

        if 'worksheets' in js:
            if len(js['worksheets']) > 0:
                if js['worksheets'][0]['cells'] is not None:
                    cells = js['worksheets'][0]['cells']
        else:
            if 'cells' in js:
                cells = js['cells']

        for cell in cells:
            if cell['cell_type'] == 'heading' or cell['cell_type'] == 'markdown':
                return markdown.markdown(''.join(cell['source'][0]).replace('#',''))
    except Exception as e:
        print(e, "\n")
        pass

    return os.path.basename(fname)

def get_notebooks():
    notebooks = []
    rel_path = "/gensim/static/notebooks/"
    for _file in os.listdir(NOTEBOOK_DIR):
        if _file.endswith(".html"):
            notebook_url = rel_path + _file
            notebook_image = notebook_url[:-5] + '.png'
            if not os.path.isfile(notebook_image[8:]):
                notebook_image = rel_path + "default.png"
            notebook_title = _file[0:-5].replace('_', ' ')
            notebook_abstract = get_abstract(os.path.abspath(os.path.join(os.path.realpath(__file__), '../../notebooks', _file.replace('.html', '.ipynb'))))
            notebooks.append({
                'url': notebook_url,
                'image': notebook_image,
                'title': notebook_title,
                'abstract': notebook_abstract})

    return notebooks
