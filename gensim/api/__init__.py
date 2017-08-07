from __future__ import print_function

import logging
import json
import os
import tarfile
import shutil
from ..utils import SaveLoad
try:
    import urllib.request as urllib
except ImportError:
    import urllib

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename="api.log", level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def download(file_name):
    url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/"
    url = url + "download/" + file_name + "/" + file_name + ".tar.gz"
    user_dir = os.path.expanduser('~')
    base_dir = os.path.join(user_dir, 'gensim-data')
    extracted_folder_dir = os.path.join(base_dir, file_name)
    if not os.path.exists(base_dir):
        logging.info("Creating {}".format(base_dir))
        os.makedirs(base_dir)
        if os.path.exists(base_dir):
            logging.info("Creation successful. Models/corpus can be accessed"
                         "via {}".format(base_dir))
        else:
            logging.error("Not able to create {d}. Make sure you have the "
                          "correct read/write permissions of the {d} or you "
                          "can try creating it manually".format(d=base_dir))
    compressed_folder_name = file_name + ".tar.gz"
    compressed_folder_dir = os.path.join(base_dir, compressed_folder_name)
    if not os.path.exists(extracted_folder_dir):
        logging.info("Downloading {}".format(file_name))
        urllib.urlretrieve(url, compressed_folder_dir)
        logging.info("{} downloaded".format(file_name))
        logging.info("Creating {}".format(extracted_folder_dir))
        os.makedirs(extracted_folder_dir)
        if os.path.exists(extracted_folder_dir):
            logging.info("Creation of {} successful"
                         ".".format(extracted_folder_dir))
            tar = tarfile.open(compressed_folder_dir)
            logging.info("Extracting files from"
                         "{}".format(extracted_folder_dir))
            tar.extractall(extracted_folder_dir)
            tar.close()
            logging.info("{} installed".format(file_name))
        else:
            logging.error("Not able to create {d}. Make sure you have the "
                          "correct read/write permissions of the {d} or you "
                          "can try creating it "
                          "manually".format(d=extracted_folder_dir))
    else:
        print("{} has already been installed".format(file_name))


def catalogue(print_list=True):
    url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/"
    url = url + "master/list.json"
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
