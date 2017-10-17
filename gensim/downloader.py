from __future__ import absolute_import
import argparse
import os
import json
import tarfile
import logging
import sys
import errno
import hashlib
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


def _create_base_dir():
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
    hash_md5 = hashlib.md5()
    with open(tar_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def info(name=None):
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
    data = info()
    corpora = data['corpora']
    models = data['models']
    if name in corpora:
        return data['corpora'][name]["checksum"]
    elif name in models:
        return data['models'][name]["checksum"]


def _download(name):
    url_data = "https://github.com/chaitaliSaini/gensim-data/releases/download/{f}/{f}.tar.gz".format(f=name)
    url_load_file = "https://github.com/chaitaliSaini/gensim-data/releases/download/{f}/__init__.py".format(f=name)
    data_folder_dir = os.path.join(base_dir, name)
    compressed_folder_name = "{f}.tar.gz".format(f=name)
    tmp_dir = tempfile.mkdtemp()
    tmp_data_folder_dir = os.path.join(tmp_dir, name)
    os.makedirs(tmp_data_folder_dir)
    if not os.path.exists(tmp_data_folder_dir):
        raise Exception(
            "Not able to create data folder in {a}. Make sure you have the correct"
            " read/write permissions for {a}".format(a=os.path.dirname(tmp_dir)))
    tmp_load_file = os.path.join(tmp_data_folder_dir, "__init__.py")
    urllib.urlretrieve(url_load_file, tmp_load_file)
    logger.info("Downloading %s", name)
    tmp_data_file = os.path.join(tmp_dir, compressed_folder_name)
    urllib.urlretrieve(url_data, tmp_data_file)
    if _calculate_md5_checksum(tmp_data_file) == _get_checksum(name):
        logger.info("%s downloaded", name)
    else:
        rmtree(tmp_dir)
        raise Exception("There was a problem in downloading the data. We recommend you to re-try.")
    tar = tarfile.open(tmp_data_file)
    tar.extractall(tmp_data_folder_dir)
    tar.close()
    os.remove(tmp_data_file)
    os.rename(tmp_data_folder_dir, data_folder_dir)
    os.rmdir(tmp_dir)


def _get_filename(name):
    data = info()
    corpora = data['corpora']
    models = data['models']
    if name in corpora:
        return data['corpora'][name]["file_name"]
    elif name in models:
        return data['models'][name]["file_name"]


def load(name, return_path=False):
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
    parser = argparse.ArgumentParser(description="Gensim console API", usage="python -m gensim.api.downloader  [-h] [-d data__name | -i data__name | -c]")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--download", metavar="data__name", nargs=1, help="To download a corpus/model : python -m gensim.downloader -d corpus/model name")
    group.add_argument("-i", "--info", metavar="data__name", nargs=1, help="To get information about a corpus/model : python -m gensim.downloader -i model/corpus name")
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
