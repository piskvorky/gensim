from __future__ import absolute_import
import argparse
import json
import os
import tarfile
import logging
import sys
import errno
import hashlib
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
data_log_file_dir = os.path.join(base_dir, 'data.json')

logging.basicConfig(
    format='%(asctime)s :%(name)s :%(levelname)s : %(message)s',
    stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('gensim.api')

if not os.path.isdir(base_dir):
    try:
        logger.info("Creating %s", base_dir)
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


def initialize_data_log_file():
    """Function for initializing the log file. Creates a json object
    for each corpus/model and stores in the log file. For eg: {"name": "text8", "status" : "None"}
    """
    data = info()
    corpora = data['gensim']['corpus']
    models = data['gensim']['model']
    json_list = []
    for corpus in corpora:
        json_object = {"name": corpus, "status": "None"}
        json_list.append(json_object)
    for model in models:
        json_object = {"name": model, "status": "None"}
        json_list.append(json_object)
    json.dump(json_list, data_log_file)
    data_log_file.close()


def update_data_log_file(dataset, status):
    """Function for updating the status of the dataset json object.

    Args:
        dataset(string): Name of the corpus/model.
        status(string): Status to be updates to i.e downloaded or installed.
    """
    jdata = json.loads(open(data_log_file_dir).read())
    for json_object in jdata:
        if json_object["name"] == dataset:
            json_object["status"] = status
    with open(data_log_file_dir, 'w') as f:
        f.write(json.dumps(jdata))


def get_data_status(dataset):
    """Function for finding the status of the dataset.

    Args:
        dataset(string): Name of the corpus/model.

    Returns:
        string: returns the current status of the corpus/model i.e None, downloaded or installed.
    """
    jdata = json.loads(open(data_log_file_dir).read())
    for json_object in jdata:
        if json_object["name"] == dataset:
            return json_object["status"]


def calculate_md5_checksum(folder_dir):
    """Function for calculating checksum of a downloaded model/corpus.

    Args:
        folder_dir(string): Path to the downloaded model.

    Returns:
        string: It returns the value for the checksum for folder_dir directory
    """
    hash_md5 = hashlib.md5()
    for filename in os.listdir(folder_dir):
        file_dir = os.path.join(folder_dir, filename)
        with open(file_dir, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


def info(dataset=None):
    """Function for retrieving the list of corpora/models, if dataset is not provided. If dataset
    is provided, then it gives detailed information about the dataset.

    Args:
        dataset(string): Name of the corpus/model.

    Returns:
        <class 'dict'>: It returns the models/corpora names with detailed information about each.
    """
    url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/list_with_filename.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    if dataset is not None:
        corpora = data['gensim']['corpus']
        models = data['gensim']['model']
        if dataset in corpora:
            logger.info("%s \n", data['gensim']['corpus'][dataset]["desc"])
        elif dataset in models:
            logger.info("%s \n", data['gensim']['model'][dataset]["desc"])
        else:
            raise Exception(
                "Incorrect model/corpus name. Choose the model/corpus from the list "
                "\n {}".format(json.dumps(data, indent=4)))
    else:
        return data


def get_checksum(dataset):
    """Function for retrieving the checksum of a corpus/model

    Args:
        dataset(string): Name of the corpus/model.

    Returns:
        string: It returns the checksum for corresponding the corpus/model.
    """
    data = info()
    corpora = data['gensim']['corpus']
    models = data['gensim']['model']
    if dataset in corpora:
        return data['gensim']['corpus'][dataset]["checksum"]
    elif dataset in models:
        return data['gensim']['model'][dataset]["checksum"]


if not os.path.isfile(data_log_file_dir):
    try:
        logger.warning("Creating %s", data_log_file_dir)
        data_log_file = open(data_log_file_dir, 'a')
        initialize_data_log_file()
    except:
        raise Exception(
            "Can't create {}. Make sure you have the read/write permissions "
            "to the directory or you can try creating the file manually"
            .format(data_log_file_dir))


def _download(dataset):
    """Function for downloading and installed dataset depending upon it's current status.

    Args:
        dataset(string): Name of the corpus/model.
    """
    url = "https://github.com/chaitaliSaini/Corpus_and_models/releases/download/{f}/{f}.tar.gz".format(f=dataset)
    data_folder_dir = os.path.join(base_dir, dataset)
    data = info()
    corpora = data['gensim']['corpus']
    models = data['gensim']['model']
    if dataset not in corpora and dataset not in models:
        raise Exception(
            "Incorect Model/corpus name. Use info() or"
            " python -m gensim.downloader -c to get a list of models/corpora"
            " available.")
    compressed_folder_name = "{f}.tar.gz".format(f=dataset)
    compressed_folder_dir = os.path.join(base_dir, compressed_folder_name)
    if get_data_status(dataset) != "downloaded":
        if not os.path.exists(data_folder_dir):
            logger.info("Creating %s", data_folder_dir)
            os.makedirs(data_folder_dir)
            if os.path.exists(data_folder_dir):
                logger.info("Creation of %s successful.", data_folder_dir)
            else:
                raise Exception(
                    "Not able to create {a}. Make sure you have the correct read/"
                    "write permissions for {b} or you can try creating it manually".
                    format(a=data_folder_dir, b=base_dir))
        logger.info("Downloading %s", dataset)
        urllib.urlretrieve(url, compressed_folder_dir)
        data_url = data_links(dataset)
        if data_url is not None:
            index = data_url.rfind("/")
            data_dir = os.path.join(data_folder_dir, data_url[index + 1:])
            urllib.urlretrieve(data_url, data_dir)
        logger.info("%s downloaded", dataset)
        update_data_log_file(dataset, status="downloaded")
    if get_data_status(dataset) != "installed":
            tar = tarfile.open(compressed_folder_dir)
            logger.info("Extracting files from %s", data_folder_dir)
            tar.extractall(data_folder_dir)
            tar.close()
            if calculate_md5_checksum(data_folder_dir) == get_checksum(dataset):
                update_data_log_file(dataset, status="installed")
                logger.info("%s installed", dataset)
            else:
                logger.error("There was a problem in installing the file. Retrying.")
                _download(dataset)


def get_filename(dataset):
    """Function of retrieving the filename of corpus/model.

    Args:
        dataset(string): Name of the corpus/model.

    Returns:
        string: Returns the filename of the model/corpus.
    """
    data = info()
    corpora = data['gensim']['corpus']
    models = data['gensim']['model']
    if dataset in corpora:
        return data['gensim']['corpus'][dataset]["filename"]
    elif dataset in models:
        return data['gensim']['model'][dataset]["filename"]


def load(dataset, return_path=False):
    """Loads the corpus/model to the memory, if return_path is False.

    Args:
        dataset(string): Name of the corpus/model.
        return_path(bool): Determines whether to return model/corpus file path.

    Returns:
        string: Returns the path to the model/corpus, if return_path is True.
    """
    file_name = get_filename(dataset)
    if file_name is None:
        raise Exception(
            "Incorrect model/corpus name. Choose the model/corpus from the list "
            "\n {}".format(json.dumps(info(), indent=4)))
    folder_dir = os.path.join(base_dir, dataset)
    file_dir = os.path.join(folder_dir, file_name)
    if not os.path.exists(folder_dir) or get_data_status(dataset) != "installed":
        _download(dataset)
    if return_path:
        return file_dir
    else:
        sys.path.insert(0, base_dir)
        module = __import__(dataset)
        data = module.load_data()
        return data


def data_links(dataset):
    """Function for retrieving the links of the models/corpus which are not stored in github releases

    Args:
        dataset(string): Name of the corpus/model.

    Returns:
        string: Returns the link of the model/corpus.
    """
    url = "https://raw.githubusercontent.com/chaitaliSaini/Corpus_and_models/master/links.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    if dataset in data['data_links']:
        return data['data_links'][dataset]['link']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gensim console API", usage="python -m gensim.api.downloader  [-h] [-d dataset_name | -i dataset_name | -c]")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--download", metavar="dataset_name", nargs=1, help="To download a corpus/model : python -m gensim -d corpus/model name")
    group.add_argument("-i", "--info", metavar="dataset_name", nargs=1, help="To get information about a corpus/model : python -m gensim -i model/corpus name")
    group.add_argument("-c", "--catalogue", help="To get the list of all models/corpus stored : python -m gensim -c", action="store_true")
    args = parser.parse_args()
    if args.download is not None:
        data_path = load(args.download[0], return_path=True)
        logger.info("Data has been installed and data path is %s", data_path)
    elif args.info is not None:
        info(dataset=args.info[0])
    elif args.catalogue is not None:
        data = info()
        logger.info("%s\n", json.dumps(data, indent=4))
