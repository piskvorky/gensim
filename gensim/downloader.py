"""
This module is an API for downloading, getting information and loading datasets/models.

Give information about available models/datasets:

>>> import gensim.downloader as api
>>>
>>> api.info()  # return dict with info about available models/datasets
>>> api.info("text8")  # return dict with info about "text8" dataset


Model example:

>>> import gensim.downloader as api
>>>
>>> model = api.load("glove-twitter-25")  # load glove vectors
>>> model.most_similar("cat")  # show words that similar to word 'cat'


Dataset example:

>>> import gensim.downloader as api
>>> from gensim.models import Word2Vec
>>>
>>> dataset = api.load("text8")  # load dataset as iterable
>>> model = Word2Vec(dataset)  # train w2v model


Also, this API available via CLI::

    python -m gensim.downloader --info <dataname> # same as api.info(dataname)
    python -m gensim.downloader --download <dataname> # same as api.load(dataname, return_path=True)

"""
from __future__ import absolute_import
import argparse
import os
import json
import logging
import sys
import errno
import hashlib
import math
import shutil
import tempfile
from functools import partial

if sys.version_info[0] == 2:
    import urllib
    from urllib2 import urlopen
else:
    import urllib.request as urllib
    from urllib.request import urlopen

user_dir = os.path.expanduser('~')
base_dir = os.path.join(user_dir, 'gensim-data')
logger = logging.getLogger('gensim.api')


def _progress(chunks_downloaded, chunk_size, total_size, part=1, total_parts=1):
    """Reporthook for :func:`urllib.urlretrieve`.

    Parameters
    ----------
    chunks_downloaded : int
        Number of chunks of data that have been downloaded.
    chunk_size : int
        Size of each chunk of data.
    total_size : int
        Total size of the dataset/model.
    part : int, optional
        Number of current part, used only if `no_parts` > 1.
    total_parts : int, optional
        Total number of parts.


    References
    ----------
    [1] https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    """
    bar_len = 50
    size_downloaded = float(chunks_downloaded * chunk_size)
    filled_len = int(math.floor((bar_len * size_downloaded) / total_size))
    percent_downloaded = round(((size_downloaded * 100) / total_size), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if total_parts == 1:
        sys.stdout.write(
            '\r[%s] %s%s %s/%sMB downloaded' % (
                bar, percent_downloaded, "%",
                round(size_downloaded / (1024 * 1024), 1),
                round(float(total_size) / (1024 * 1024), 1))
        )
        sys.stdout.flush()
    else:
        sys.stdout.write(
            '\r Part %s/%s [%s] %s%s %s/%sMB downloaded' % (
                part + 1, total_parts, bar, percent_downloaded, "%",
                round(size_downloaded / (1024 * 1024), 1),
                round(float(total_size) / (1024 * 1024), 1))
        )
        sys.stdout.flush()


def _create_base_dir():
    """Create the gensim-data directory in home directory, if it has not been already created.

    Raises
    ------
    Exception
        An exception is raised when read/write permissions are not available or a file named gensim-data
        already exists in the home directory.

    """
    if not os.path.isdir(base_dir):
        try:
            logger.info("Creating %s", base_dir)
            os.makedirs(base_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception(
                    "Not able to create folder gensim-data in {}. File gensim-data "
                    "exists in the direcory already.".format(user_dir)
                )
            else:
                raise Exception(
                    "Can't create {}. Make sure you have the read/write permissions "
                    "to the directory or you can try creating the folder manually"
                    .format(base_dir)
                )


def _calculate_md5_checksum(fname):
    """Calculate the checksum of the file, exactly same as md5-sum linux util.

    Parameters
    ----------
    fname : str
        Path to the file.

    Returns
    -------
    str
        MD5-hash of file names as `fname`.

    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def info(name=None):
    """Provide the information related to model/dataset.

    Parameters
    ----------
    name : str, optional
        Name of model/dataset.

    Returns
    -------
    dict
        Detailed information about one or all models/datasets.
        If name is specified, return full information about concrete dataset/model,
        otherwise, return information about all available datasets/models.

    Raises
    ------
    Exception
        If name that has been passed is incorrect.

    Examples
    --------
    >>> import gensim.downloader as api
    >>> api.info("text8")  # retrieve information about text8 dataset
    {u'checksum': u'68799af40b6bda07dfa47a32612e5364',
     u'description': u'Cleaned small sample from wikipedia',
     u'file_name': u'text8.gz',
     u'parts': 1,
     u'source': u'http://mattmahoney.net/dc/text8.zip'}
    >>>
    >>> api.info()  # retrieve information about all available datasets and models

    """
    url = "https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list.json"
    information = json.loads(urlopen(url).read().decode("utf-8"))

    if name is not None:
        corpora = information['corpora']
        models = information['models']
        if name in corpora:
            return information['corpora'][name]
        elif name in models:
            return information['models'][name]
        else:
            raise Exception("Incorrect model/corpus name")
    else:
        return information


def _get_checksum(name, part=None):
    """Retrieve the checksum of the model/dataset from gensim-data repository.

    Parameters
    ----------
    name : str
        Dataset/model name.
    part : int, optional
        Number of part (for multipart data only).

    Returns
    -------
    str
        Retrieved checksum of dataset/model.

    """
    information = info()
    corpora = information['corpora']
    models = information['models']
    if part is None:
        if name in corpora:
            return information['corpora'][name]["checksum"]
        elif name in models:
            return information['models'][name]["checksum"]
    else:
        if name in corpora:
            return information['corpora'][name]["checksum-{}".format(part)]
        elif name in models:
            return information['models'][name]["checksum-{}".format(part)]


def _get_parts(name):
    """Retrieve the number of parts in which dataset/model has been split.

    Parameters
    ----------
    name: str
        Dataset/model name.

    Returns
    -------
    int
        Number of parts in which dataset/model has been split.

    """
    information = info()
    corpora = information['corpora']
    models = information['models']
    if name in corpora:
        return information['corpora'][name]["parts"]
    elif name in models:
        return information['models'][name]["parts"]


def _download(name):
    """Download and extract the dataset/model.

    Parameters
    ----------
    name: str
        Dataset/model name which has to be downloaded.

    Raises
    ------
    Exception
        If md5sum on client and in repo are different.

    """
    url_load_file = "https://github.com/RaRe-Technologies/gensim-data/releases/download/{f}/__init__.py".format(f=name)
    data_folder_dir = os.path.join(base_dir, name)
    tmp_dir = tempfile.mkdtemp()
    init_path = os.path.join(tmp_dir, "__init__.py")
    urllib.urlretrieve(url_load_file, init_path)
    total_parts = _get_parts(name)
    if total_parts > 1:
        concatenated_folder_name = "{f}.gz".format(f=name)
        concatenated_folder_dir = os.path.join(tmp_dir, concatenated_folder_name)
        for part in range(0, total_parts):
            url_data = \
                "https://github.com/RaRe-Technologies/gensim-data/releases/download/{f}/{f}.gz_0{p}" \
                .format(f=name, p=part)

            fname = "{f}.gz_0{p}".format(f=name, p=part)
            dst_path = os.path.join(tmp_dir, fname)
            urllib.urlretrieve(
                url_data, dst_path,
                reporthook=partial(_progress, part=part, total_parts=total_parts)
            )
            if _calculate_md5_checksum(dst_path) == _get_checksum(name, part):
                sys.stdout.write("\n")
                sys.stdout.flush()
                logger.info("Part %s/%s downloaded", part + 1, total_parts)
            else:
                shutil.rmtree(tmp_dir)
                raise Exception("Checksum comparison failed, try again")
        with open(concatenated_folder_dir, 'wb') as wfp:
            for part in range(0, total_parts):
                part_path = os.path.join(tmp_dir, "{f}.gz_0{p}".format(f=name, p=part))
                with open(part_path, "rb") as rfp:
                    shutil.copyfileobj(rfp, wfp)
                os.remove(part_path)
        os.rename(tmp_dir, data_folder_dir)
    else:
        url_data = "https://github.com/RaRe-Technologies/gensim-data/releases/download/{f}/{f}.gz".format(f=name)
        fname = "{f}.gz".format(f=name)
        dst_path = os.path.join(tmp_dir, fname)
        urllib.urlretrieve(url_data, dst_path, reporthook=_progress)
        if _calculate_md5_checksum(dst_path) == _get_checksum(name):
            sys.stdout.write("\n")
            sys.stdout.flush()
            logger.info("%s downloaded", name)
        else:
            shutil.rmtree(tmp_dir)
            raise Exception("Checksum comparison failed, try again")
        os.rename(tmp_dir, data_folder_dir)


def _get_filename(name):
    """Retrieve the filename of the dataset/model.

    Parameters
    ----------
    name: str
        Name of dataset/model.

    Returns
    -------
    str:
        Filename of the dataset/model.

    """
    information = info()
    corpora = information['corpora']
    models = information['models']
    if name in corpora:
        return information['corpora'][name]["file_name"]
    elif name in models:
        return information['models'][name]["file_name"]


def load(name, return_path=False):
    """Download (if needed) dataset/model and load it to memory (managed by return_path).

    Parameters
    ----------
    name: str
        Name of the model/dataset.
    return_path: bool, optional
        If True, return full path to file, otherwise, return loaded model / iterable dataset.

    Returns
    -------
    Model
        Requested model, if `name` is model and `return_path` == False.
    Dataset (iterable)
        Requested dataset, if `name` is dataset and `return_path` == False.
    str
        Path to file with dataset / model, only when `return_path` == True.

    Raises
    ------
    Exception
        Raised if `name` is incorrect.

    Examples
    --------
    Model example:

    >>> import gensim.downloader as api
    >>>
    >>> model = api.load("glove-twitter-25")  # load glove vectors
    >>> model.most_similar("cat")  # show words that similar to word 'cat'

    Dataset example:

    >>> import gensim.downloader as api
    >>>
    >>> wiki = api.load("wiki-en")  # load extracted Wikipedia dump, around 6 Gb
    >>> for article in wiki:  # iterate over all wiki script
    >>>     ...

    Download only example
    >>> import gensim.downloader as api
    >>>
    >>> print(api.load("wiki-en", return_path=True))  # output: /home/user/gensim-data/wiki-en/wiki-en.gz

    """
    _create_base_dir()
    file_name = _get_filename(name)
    if file_name is None:
        raise Exception(
            "Incorrect model/corpus name. Choose the model/corpus from the list "
            "\n {}".format(json.dumps(info(), indent=4))
        )
    folder_dir = os.path.join(base_dir, name)
    path = os.path.join(folder_dir, file_name)
    if not os.path.exists(folder_dir):
        _download(name)

    if return_path:
        return path
    else:
        sys.path.insert(0, base_dir)
        module = __import__(name)
        return module.load_data()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s :%(name)s :%(levelname)s :%(message)s', stream=sys.stdout, level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description="Gensim console API",
        usage="python -m gensim.api.downloader  [-h] [-d data_name | -i data_name | -c]"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-d", "--download", metavar="data_name", nargs=1,
        help="To download a corpus/model : python -m gensim.downloader -d <dataname>"
    )

    full_information = 1
    group.add_argument(
        "-i", "--info", metavar="data_name", nargs='?', const=full_information,
        help="To get information about a corpus/model : python -m gensim.downloader -i <dataname>"
    )

    args = parser.parse_args()
    if args.download is not None:
        data_path = load(args.download[0], return_path=True)
        logger.info("Data has been installed and data path is %s", data_path)
    elif args.info is not None:
        output = info() if (args.info == full_information) else info(name=args.info)
        print(json.dumps(info, indent=4))
