import subprocess
import json
import sys
import os
import pip
import importlib
from pathlib import Path
import os
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


def download(file):
    url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/download/"
    data = catalogue(False)
    corpuses = data['gensim']['corpus']
    models = data['gensim']['model']
    if file not in corpuses and file not in models:
        print("Incorrect corpus/model name.")
    else:
        url = url+file+"/"+file+".tar.gz"
        print("Downloading {m}".format(m=file))
        subprocess.call(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', url],
            env=os.environ.copy())


def catalogue(print_list=True):
    url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/list.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    if print_list == 1:
        data = json.loads(data)
        corpuses = data['gensim']['corpus']
        models = data['gensim']['model']
        print("Corpuses available : ")
        for corpus in corpuses:
            print(corpus)
        print("")
        print("Models available : ")
        for model in models:
            print(model)
    else:
        return json.loads(data)


def link(file_name, shortcut_name):
    packages = pip.get_installed_distributions()
    package_is_installed = 0
    for package in packages:
        if package.project_name == file_name:
            package_is_installed = 1
            break
    if package_is_installed == 0:
        print("The model/corpus {f} has not been installed".format(f=file_name))
        print("For installing use: python -m gensim -d {f}".format(f=file_name))
        sys.exit(0)
    package = importlib.import_module(file_name)
    package_path = Path(package.__file__).parent.parent
    package_path = package_path / file_name / file_name
    gensim_path = importlib.import_module("gensim")
    gensim_path = Path(gensim_path.__file__).parent
    shortcut_path = gensim_path / 'data' / shortcut_name
    if os.path.exists(str(shortcut_path)):
        print("This shortcut link already exists.")
        sys.exit(0)
    try:
        os.symlink(str(package_path), str(shortcut_path))
    except:
        print("Shortcut creation failed in gensim/data.")
        sys.exit(0)
    print("Shortcut creation successful. The model/corpus can now be found"
          " in gensim/data.")


def info(file_name):
    data = catalogue(False)
    corpuses = data['gensim']['corpus']
    models = data['gensim']['model']
    if file_name in corpuses:
        print(data['gensim']['corpus'][file_name])
    elif file_name in models:
        print(data['gensim']['model'][file_name])
    else:
        print("Incorrect model/corpus name.")
