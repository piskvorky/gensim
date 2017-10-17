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
data_log_file_path = os.path.join(base_dir, 'data.json')
logger = logging.getLogger('gensim.api')


def get_data_list():
    """Function getting the list of all datasets/models.

    Returns:
        list: returns a list of datasets/models avalible for installation.
    """
    url = "https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list_with_filename.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    data_names = []
    corpora = data['corpus']
    models = data['model']
    for corpus in corpora:
        data_names.append(corpus)
    for model in models:
        data_names.append(model)
    return data_names


def get_data_name(data_):
    """Returns a name for the dataset/model as to download a dataset/model user can alternate names too.

    Args:
        data_(string): Name of the corpus/model.

    Returns:
        data_: returns the name for dataset/model
    """
    url = "https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/alternate_names.json"
    response = urlopen(url)
    alternate_names_json = response.read().decode("utf-8")
    alternate_names_json = json.loads(alternate_names_json)
    data_names = get_data_list()
    for data_name in data_names:
        alternate_data_names = alternate_names_json[data_name]
        if data_ in alternate_data_names:
            return data_name


def initialize_data_log_file():
    """Function for initializing the log file. Creates a json object
    for each corpus/model and stores in the log file. For eg: {"name": "text8", "status" : "None"}
    """
    data = info()
    corpora = data['corpus']
    models = data['model']
    json_list = []
    for corpus in corpora:
        json_object = {"name": corpus, "status": "None"}
        json_list.append(json_object)
    for model in models:
        json_object = {"name": model, "status": "None"}
        json_list.append(json_object)

    with open(data_log_file_path, 'w') as f:
        f.write(json.dumps(json_list))


def create_files():
    """Function for creating the directory for storing corpora and models, and to create a json log file.
    """
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

    if not os.path.isfile(data_log_file_path):
        try:
            logger.warning("Creating %s", data_log_file_path)
            with open(data_log_file_path, 'w+'):
                pass
            initialize_data_log_file()
        except:
            raise Exception(
                "Can't create {}. Make sure you have the read/write permissions "
                "to the directory or you can try creating the file manually"
                .format(data_log_file_path))


def update_data_log_file(data_, status):
    """Function for updating the status of the data_ json object.

    Args:
        data_(string): Name of the corpus/model.
        status(string): Status to be updates to i.e downloaded or installed.
    """
    with open(data_log_file_path, 'r') as f:
        jdata = json.load(f)
        for json_object in jdata:
            if json_object["name"] == data_:
                json_object["status"] = status
    with open(data_log_file_path, 'w+') as f:
        f.write(json.dumps(jdata))


def get_data_status(data_):
    """Function for finding the status of the data_.

    Args:
        data_(string): Name of the corpus/model.

    Returns:
        string: returns the current status of the corpus/model i.e None, downloaded or installed.
    """
    with open(data_log_file_path, 'r') as f:
        jdata = json.load(f)
    for json_object in jdata:
        if json_object["name"] == data_:
            return json_object["status"]


def calculate_md5_checksum(folder_dir, tar_file=None):
    """Function for calculating checksum of a downloaded or installed model/corpus.

    Args:
        folder_dir(string): Path to the model/corpus folder.(contains model/corpus if proxied)
        tar_file(string): Path to the dowloaded tar file. Tar file contains __init__.py file and the model/corpus(if it is stored in github releases)

    Returns:
        string: It returns the value for the checksum for folder_dir directory and the tar file.
    """
    hash_md5 = hashlib.md5()
    for filename in os.listdir(folder_dir):
        file_dir = os.path.join(folder_dir, filename)
        with open(file_dir, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    if tar_file is not None:
        with open(tar_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


def info(data_=None):
    """Function for retrieving the list of corpora/models, if data name is not provided. If data name
    is provided, then it gives detailed information about the data.

    Args:
        data_(string): Name of the corpus/model.

    Returns:
        <class 'dict'>: It returns the models/corpora names with detailed information about each.
    """
    url = "https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list_with_filename.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    if data_ is not None:
        data_ = get_data_name(data_)
        corpora = data['corpus']
        models = data['model']
        if data_ in corpora:
            logger.info("%s \n", data['corpus'][data_]["desc"])
            return data['corpus'][data_]["desc"]
        elif data_ in models:
            logger.info("%s \n", data['model'][data_]["desc"])
            return data['model'][data_]["desc"]
        else:
            raise Exception(
                "Incorrect model/corpus name. Choose the model/corpus from the list "
                "\n {}".format(json.dumps(data, indent=4)))
    else:
        return data


def get_checksum(data_, status):
    """Function for retrieving the checksum of a corpus/model

    Args:
        data_(string): Name of the corpus/model.

    Returns:
        string: It returns the checksum for corresponding the corpus/model.
    """
    key = "checksum_after_" + status
    data = info()
    corpora = data['corpus']
    models = data['model']
    if data_ in corpora:
        return data['corpus'][data_][key]
    elif data_ in models:
        return data['model'][data_][key]


def _download(data_):
    """Function for downloading and installed corpus/model depending upon it's current status.

    Args:
        data_(string): Name of the corpus/model.
    """
    url = "https://github.com/RaRe-Technologies/gensim-data/releases/download/{f}/{f}.tar.gz".format(f=data_)
    data_folder_dir = os.path.join(base_dir, data_)
    data = info()
    corpora = data['corpus']
    models = data['model']
    if data_ not in corpora and data_ not in models:
        raise Exception(
            "Incorect Model/corpus name. Use info() or"
            " python -m gensim.downloader -c to get a list of models/corpora"
            " available.")
    compressed_folder_name = "{f}.tar.gz".format(f=data_)
    compressed_folder_dir = os.path.join(base_dir, compressed_folder_name)
    if get_data_status(data_) != "downloaded":
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
        logger.info("Downloading %s", data_)
        urllib.urlretrieve(url, compressed_folder_dir)
        data_url = data_links(data_)
        if data_url is not None:
            index = data_url.rfind("/")
            data_dir = os.path.join(data_folder_dir, data_url[index + 1:])
            urllib.urlretrieve(data_url, data_dir)
        if calculate_md5_checksum(data_folder_dir, compressed_folder_dir) == get_checksum(data_, "download"):
            logger.info("%s downloaded", data_)
            update_data_log_file(data_, status="downloaded")
        else:
            logger.error("There was a problem in downloading the data. Retrying.")
            _download(data_)

    if get_data_status(data_) != "installed":
            tar = tarfile.open(compressed_folder_dir)
            logger.info("Extracting files from %s", data_folder_dir)
            tar.extractall(data_folder_dir)
            tar.close()
            if calculate_md5_checksum(data_folder_dir) == get_checksum(data_, "installation"):
                update_data_log_file(data_, status="installed")
                logger.info("%s installed", data_)
            else:
                logger.error("There was a problem in installing the dataset/model. Retrying.")
                _download(data_)


def get_filename(data_):
    """Function of retrieving the filename of corpus/model.

    Args:
        data_(string): Name of the corpus/model.

    Returns:
        string: Returns the filename of the model/corpus.
    """
    data = info()
    corpora = data['corpus']
    models = data['model']
    if data_ in corpora:
        return data['corpus'][data_]["filename"]
    elif data_ in models:
        return data['model'][data_]["filename"]


def load(data_, return_path=False):
    """Loads the corpus/model to the memory, if return_path is False.

    Args:
        data_(string): Name of the corpus/model.
        return_path(bool): Determines whether to return model/corpus file path.

    Returns:
        string: Returns the path to the model/corpus, if return_path is True.
    """
    data_ = get_data_name(data_)
    create_files()
    file_name = get_filename(data_)
    if file_name is None:
        raise Exception(
            "Incorrect model/corpus name. Choose the model/corpus from the list "
            "\n {}".format(json.dumps(info(), indent=4)))
    folder_dir = os.path.join(base_dir, data_)
    file_dir = os.path.join(folder_dir, file_name)
    if not os.path.exists(folder_dir) or get_data_status(data_) != "installed":
        _download(data_)

    if return_path:
        return file_dir
    else:
        sys.path.insert(0, base_dir)
        module = __import__(data_)
        data = module.load_data()
        return data


def data_links(data_):
    """Function for retrieving the links of the models/corpus which are not stored in github releases

    Args:
        data_(string): Name of the corpus/model.

    Returns:
        string: Returns the link of the model/corpus.
    """
    url = "https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/links.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    if data_ in data['data_links']:
        return data['data_links'][data_]['link']


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s :%(name)s :%(levelname)s :%(message)s', stream=sys.stdout, level=logging.INFO)
    parser = argparse.ArgumentParser(description="Gensim console API", usage="python -m gensim.api.downloader  [-h] [-d data__name | -i data__name | -c]")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--download", metavar="data__name", nargs=1, help="To download a corpus/model : python -m gensim -d corpus/model name")
    group.add_argument("-i", "--info", metavar="data__name", nargs=1, help="To get information about a corpus/model : python -m gensim -i model/corpus name")
    group.add_argument("-c", "--catalogue", help="To get the list of all models/corpus stored : python -m gensim -c", action="store_true")
    args = parser.parse_args()
    if args.download is not None:
        data_path = load(args.download[0], return_path=True)
        logger.info("Data has been installed and data path is %s", data_path)
    elif args.info is not None:
        info(data_=args.info[0])
    elif args.catalogue is not None:
        data = info()
        logger.info("%s\n", json.dumps(data, indent=4))
