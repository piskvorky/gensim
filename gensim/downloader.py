"""This module is an API for downloading, getting information and loading datasets/models."""
from __future__ import absolute_import
import argparse
import os
import json
import tarfile
import logging
import sys
import errno
import hashlib
import math
from shutil import rmtree
import tempfile
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
logger = logging.getLogger('gensim.api')


def _progress(chunks_downloaded, chunk_size, total_size):
    """Create and update the _progress bar.

    Parameters
    ----------
    chunks_downloaded : int
        Number of chunks of data that have been downloaded
    chunks_size : int
        Size of each chunk of data.
    total_size : int
        Total size of the dataset/model

    References
    ----------
    [1] https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    """
    bar_len = 50
    size_downloaded = float(chunks_downloaded * chunk_size)
    filled_len = int(math.floor((bar_len * size_downloaded) / total_size))
    percent_downloaded = round(((size_downloaded * 100) / total_size), 1) 
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(
        '[%s] %s%s %s/%sMB downloaded\r' % (bar, percent_downloaded, "%",
        round(size_downloaded / (1024 * 1024), 1),
        round(float(total_size) / (1024 * 1024), 1)))
    sys.stdout.flush()


def _create_base_dir():
    """Create the gensim-data directory in home directory, if it has not been already created.
    Raises
    ------
    File Exists Error
        Give this exception when a file gensim-data already exists in the home directory.
    Exception
        An exception is raised when read/write permissions are not available

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


def _calculate_md5_checksum(tar_file):
    """Calculate the checksum of the given tar.gz file.
    Parameters
    ----------
    tar_file : str
        Path to the tar.gz file that contains the downloaded dataset/model.

    Returns
    -------
    str
        Return the calculated checksum of the tsr.gz file.

    """
    hash_md5 = hashlib.md5()
    with open(tar_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def info(name=None):
    """Return the information related to model/dataset.

    If name is supplied, then information related to the given dataset/model will
    be returned. Otherwise detailed information of all model/datasets will be returned.

    Parameters
    ----------
    name : str or None, optional
        str stores the data name.
    Returns
    -------
    dict
        Return detailed information about all models/datasets if name is not
        provided. Otherwise return detailed informtiona of the specific model/dataset

    Raises
    ------
    Exception
        If name that has been passed is incorrect, an exception is raised.

    """
    url = "https://raw.githubusercontent.com/chaitaliSaini/gensim-data/master/list.json"
    response = urlopen(url)
    data = response.read().decode("utf-8")
    data = json.loads(data)
    if name is not None:
        corpora = data['corpora']
        models = data['models']
        if name in corpora:
            logger.info("%s \n", json.dumps(data['corpora'][name], indent=4))
            return data['corpora'][name]
        elif name in models:
            logger.info("%s \n", json.dumps(data['corpora'][name], indent=4))
            return data['models'][name]
        else:
            raise Exception(
                "Incorrect model/corpus name. Choose the model/corpus from the list "
                "\n {}".format(json.dumps(data, indent=4)))
    else:
        return data


def _get_checksum(name):
    """Retrieve the checksum of the model/dataset.
    This is compared to the checksum of the downloaded model/dataset in _download function.

    Parameters
    ----------
    name: str
        Dataset/model name

    Returns
    -------
    str
        Retrieved checksum of dataset/model

    """
    data = info()
    corpora = data['corpora']
    models = data['models']
    if name in corpora:
        return data['corpora'][name]["checksum"]
    elif name in models:
        return data['models'][name]["checksum"]


def _download(name):
    """Download and extract the dataset/model

    Parameters
    ----------
    name: str
        Dataset/model name which has to be downloaded

    Raises
    ------
    Exception
        If checksum of dowloaded data does not match the retrieved checksum,
        then downloaded has not been donw properly.

    """
    url_data = "https://github.com/chaitaliSaini/gensim-data/releases/download/{f}/{f}.tar.gz".format(f=name)
    url_load_file = "https://github.com/chaitaliSaini/gensim-data/releases/download/{f}/__init__.py".format(f=name)
    data_folder_dir = os.path.join(base_dir, name)
    compressed_folder_name = "{f}.tar.gz".format(f=name)
    tmp_dir = tempfile.mkdtemp()
    tmp_load_file = os.path.join(tmp_dir, "__init__.py")
    urllib.urlretrieve(url_load_file, tmp_load_file)
    logger.info("Downloading %s", name)
    tmp_data_file = os.path.join(tmp_dir, compressed_folder_name)
    urllib.urlretrieve(url_data, tmp_data_file, reporthook=_progress)
    if _calculate_md5_checksum(tmp_data_file) == _get_checksum(name):
        logger.info("%s downloaded", name)
    else:
        rmtree(tmp_dir)
        raise Exception("There was a problem in downloading the data. We recommend you to re-try.")
    with tarfile.open(tmp_data_file, 'r') as tar:
        tar.extractall(tmp_dir)
    os.remove(tmp_data_file)
    os.rename(tmp_dir, data_folder_dir)


def _get_filename(name):
    """Retrive the filename(or foldername, in case of some datasets) of the dataset/model.

    Parameters
    ----------
    name: str
        Dataset/model of which filename is retrieved.

    Returns
    -------
    str:
        Filename(or foldername, in case of some datasets) of the dataset/model.

    """
    data = info()
    corpora = data['corpora']
    models = data['models']
    if name in corpora:
        return data['corpora'][name]["file_name"]
    elif name in models:
        return data['models'][name]["file_name"]


def load(name, return_path=False):
    """For models, if return_path is False, then load model to memory. Otherwise, return the path to the model.
    For datasets, return path to the dataset in both cases.

    Parameters
    ----------
    name: str
        Name of the model/dataset
    return_path: False or True, optional

    Returns
    -------
    data:
        Load model to memory
    data_dir: str
        Return path of dataset/model.

    """
    _create_base_dir()
    file_name = _get_filename(name)
    if file_name is None:
        raise Exception(
            "Incorrect model/corpus name. Choose the model/corpus from the list "
            "\n {}".format(json.dumps(info(), indent=4)))
    folder_dir = os.path.join(base_dir, name)
    data_dir = os.path.join(folder_dir, file_name)
    if not os.path.exists(folder_dir):
        _download(name)

    if return_path:
        return data_dir
    else:
        sys.path.insert(0, base_dir)
        module = __import__(name)
        data = module.load_data()
        return data


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s :%(name)s :%(levelname)s :%(message)s', stream=sys.stdout, level=logging.INFO)
    parser = argparse.ArgumentParser(description="Gensim console API", usage="python -m gensim.api.downloader  [-h] [-d data_name | -i data_name | -c]")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--download", metavar="data_name", nargs=1, help="To download a corpus/model : python -m gensim.downloader -d corpus/model name")
    group.add_argument("-i", "--info", metavar="data_name", nargs=1, help="To get information about a corpus/model : python -m gensim.downloader -i model/corpus name")
    group.add_argument("-c", "--catalogue", help="To get the list of all models/corpus stored : python -m gensim.downloader -c", action="store_true")
    args = parser.parse_args()
    if args.download is not None:
        data_path = load(args.download[0], return_path=True)
        logger.info("Data has been installed and data path is %s", data_path)
    elif args.info is not None:
        info(name=args.info[0])
    elif args.catalogue is not None:
        data = info()
        logger.info("%s\n", json.dumps(data, indent=4))
