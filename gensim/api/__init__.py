from __future__ import print_function
from __future__ import absolute_import
import json
import os
import tarfile
import logging
import sys
import errno
try:
    import urllib.request as urllib
except ImportError:
    import urllib

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


user_dir = os.path.expanduser('~')
base_dir = os.path.join(user_dir, 'gensim-data')
log_file_dir = os.path.join(base_dir, 'api.log')
if not os.path.isdir(base_dir):
    try:
        os.makedirs(base_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise Exception(
                "Not able to create folder gensim-data in {}. File gensim-data "
                "exists in the direcory already.".format(user_dir))
        else:
            raise Exception(
                "Can't create {}. Make sure you have the read/write permissions "
                "to the directory or you can try creating the folder manually"
                .format(base_dir))

logging.basicConfig(
    format='%(asctime)s :%(name)s :%(levelname)s : %(message)s',
    filename=log_file_dir, level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger = logging.getLogger('gensim.api')
logger.addHandler(console)


def download(dataset):
    url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/download/{f}/{f}.tar.gz".format(f=dataset)
    data_folder_dir = os.path.join(base_dir, dataset)
    data = catalogue(print_list=False)
    corpuses = data['gensim']['corpus']
    models = data['gensim']['model']
    if dataset not in corpuses and dataset not in models:
        logger.error(
            "Incorect Model/corpus name. Use catalogue(print_list=TRUE) or"
            " python -m gensim -c to get a list of models/corpuses"
            " available.")
        sys.exit(0)
    compressed_folder_name = "{f}.tar.gz".format(f=dataset)
    compressed_folder_dir = os.path.join(base_dir, compressed_folder_name)
    is_installed = False
    is_downloaded = False
    installed_message = "{f} installed".format(f=dataset)
    downloaded_message = "{f} downloaded".format(f=dataset)
    if os.path.exists(data_folder_dir):
        log_file_dir = os.path.join(base_dir, 'api.log')
        with open(log_file_dir) as f:
            f = f.readlines()
        for line in f:
            if installed_message in line:
                print("{} has already been installed".format(dataset))
                is_installed = True
                sys.exit(0)
    if os.path.exists(data_folder_dir) and not is_installed:
        for line in f:
            if downloaded_message in line:
                is_downloaded = True
                break
    if not is_downloaded:
        if not os.path.exists(data_folder_dir):
            logger.info("Creating %s", data_folder_dir)
            os.makedirs(data_folder_dir)
            if os.path.exists(data_folder_dir):
                logger.info("Creation of %s successful.", data_folder_dir)
            else:
                logger.error(
                    "Not able to create %s. Make sure you have the correct read/"
                    "write permissions for %s or you can try creating it manually",
                    data_folder_dir, base_dir)
                sys.exit(0)
        logger.info("Downloading %s", dataset)
        urllib.urlretrieve(url, compressed_folder_dir)
        data_url = data_links(dataset)
        if data_url is not None:
            index = data_url.rfind("/")
            data_dir = os.path.join(data_folder_dir, data_url[index + 1:])
            urllib.urlretrieve(data_url, data_dir)
        logger.info("%s downloaded", dataset)
    if not is_installed:
            tar = tarfile.open(compressed_folder_dir)
            logger.info("Extracting files from %s", data_folder_dir)
            tar.extractall(data_folder_dir)
            tar.close()
            logger.info("%s installed", dataset)


def catalogue(print_list=False):
    url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/list_with_filename.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    if print_list:
        corpuses = data['gensim']['corpus']
        models = data['gensim']['model']
        print("Corpuses available : ")
        for corpus in corpuses:
            print(corpus)
        print("")
        print("Models available : ")
        for model in models:
            print(model)
    return data


def info(dataset):
    data = catalogue()
    corpuses = data['gensim']['corpus']
    models = data['gensim']['model']
    if dataset in corpuses:
        print(data['gensim']['corpus'][dataset]["desc"])
    elif dataset in models:
        print(data['gensim']['model'][dataset]["desc"])
    else:
        catalogue(print_list=True)
        raise Exception(
            "Incorrect model/corpus name. Choose the model/corpus from the list "
            "above.")


def get_filename(dataset):
    data = catalogue()
    corpuses = data['gensim']['corpus']
    models = data['gensim']['model']
    if dataset in corpuses:
        return data['gensim']['corpus'][dataset]["filename"]
    elif dataset in models:
        return data['gensim']['model'][dataset]["filename"]


def load(dataset, return_path=False):
    file_name = get_filename(dataset)
    folder_dir = os.path.join(base_dir, dataset)
    file_dir = os.path.join(folder_dir, file_name)
    if not os.path.exists(folder_dir):
        raise Exception(
            "Incorrect model/corpus name. Use catalogue(print_list=True) to get a list of "
            "avalible models/corpus. If the model/corpus name you entered is"
            " in the catalogue, then please download the model/corpus by "
            "calling download('{f}') function".format(f=dataset))
    elif return_path:
        return file_dir
    else:
        sys.path.insert(0, base_dir)
        module = __import__(dataset)
        data = module.load_data()
        return data


def data_links(dataset):
    url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/links.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    if dataset in data['data_links']:
        return data['data_links'][dataset]['link']
