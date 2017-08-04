from __future__ import print_function
import json
import sys
import os
import six
import tarfile
try:
    import urllib.request as urllib
except ImportError:
    import urllib

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


def download(file):
    url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/"
    url = url+"download/"+file+"/"+file+".tar.gz"
    user_dir = os.path.expanduser('~')
    base_dir = os.path.join(user_dir, 'gensim-data')
    extracted_folder_dir = os.path.join(base_dir, file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    compressed_folder_name = file+".tar.gz"
    compressed_folder_dir = os.path.join(base_dir, compressed_folder_name)
    urllib.urlretrieve(url, compressed_folder_dir)
    if not os.path.exists(extracted_folder_dir):
        os.makedirs(extracted_folder_dir)
    tar = tarfile.open(compressed_folder_dir)
    tar.extractall(extracted_folder_dir)
    tar.close()


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
