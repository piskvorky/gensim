from __future__ import print_function
from __future__ import absolute_import
import json
import os
import tarfile
import shutil
import logging
import sys
import importlib
try:
    import urllib.request as urllib
except ImportError:
    import urllib

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

logger = logging.getLogger('gensim.api')


def download(file_name):
    url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/"
    url = url + "download/" + file_name + "/" + file_name + ".tar.gz"
    user_dir = os.path.expanduser('~')
    base_dir = os.path.join(user_dir, 'gensim-data')
    extracted_folder_dir = os.path.join(base_dir, file_name)
    if not os.path.exists(base_dir):
        logger.info("Creating {}".format(base_dir))
        os.makedirs(base_dir)
        if os.path.exists(base_dir):
            logger.info(
                "Creation successful. Models/corpus can be accessed"
                "via {}".format(base_dir))
        else:
            logger.error(
                "Not able to create {d}. Make sure you have the "
                "correct read/write permissions of the {d} or you "
                "can try creating it manually".format(d=base_dir))
    compressed_folder_name = file_name + ".tar.gz"
    compressed_folder_dir = os.path.join(base_dir, compressed_folder_name)
    if not os.path.exists(extracted_folder_dir):
        logger.info("Downloading {}".format(file_name))
        urllib.urlretrieve(url, compressed_folder_dir)
        logger.info("{} downloaded".format(file_name))
        logger.info("Creating {}".format(extracted_folder_dir))
        os.makedirs(extracted_folder_dir)
        if os.path.exists(extracted_folder_dir):
            logger.info(
                "Creation of {} successful"
                ".".format(extracted_folder_dir))
            tar = tarfile.open(compressed_folder_dir)
            logger.info(
                "Extracting files from"
                "{}".format(extracted_folder_dir))
            tar.extractall(extracted_folder_dir)
            tar.close()
            logger.info("{} installed".format(file_name))
        else:
            logger.error(
                "Not able to create {d}. Make sure you have the "
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
    if print_list:
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


def load(file_name, return_path=False):
    user_dir = os.path.expanduser('~')
    base_dir = os.path.join(user_dir, 'gensim-data')
    folder_dir = os.path.join(base_dir, file_name)
    if not os.path.exists(folder_dir):
        print(
            "Incorrect model/corpus name. Use catalogue() to get a list of "
            "avalible models/corpus. If the model/corpus name you entered is"
            " in the catalogue, then please download the model/corpus by "
            "calling download({f}) function".format(f=file_name))
    elif return_path:
        return folder_dir
    else:
        sys.path.insert(0, base_dir)
        module = __import__(file_name)
        data = module.load_data()
        return data
