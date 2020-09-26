"""
This module is an API for downloading, getting information and loading datasets/models.

See `RaRe-Technologies/gensim-data <https://github.com/RaRe-Technologies/gensim-data>`_ repo
for more information about models/datasets/how-to-add-new/etc.

Give information about available models/datasets:

.. sourcecode:: pycon

    >>> import gensim.downloader as api
    >>>
    >>> api.info()  # return dict with info about available models/datasets
    >>> api.info("text8")  # return dict with info about "text8" dataset


Model example:

.. sourcecode:: pycon

    >>> import gensim.downloader as api
    >>>
    >>> model = api.load("glove-twitter-25")  # load glove vectors
    >>> model.most_similar("cat")  # show words that similar to word 'cat'


Dataset example:

.. sourcecode:: pycon

    >>> import gensim.downloader as api
    >>> from gensim.models import Word2Vec
    >>>
    >>> dataset = api.load("text8")  # load dataset as iterable
    >>> model = Word2Vec(dataset)  # train w2v model


Also, this API available via CLI::

    python -m gensim.downloader --info <dataname> # same as api.info(dataname)
    python -m gensim.downloader --info name # same as api.info(name_only=True)
    python -m gensim.downloader --download <dataname> # same as api.load(dataname, return_path=True)

You may specify the local subdirectory for saving gensim data using the
GENSIM_DATA_DIR environment variable.  For example:

    $ export GENSIM_DATA_DIR=/tmp/gensim-data
    $ python -m gensim.downloader --download <dataname>

By default, this subdirectory is ~/gensim-data.

"""

from __future__ import absolute_import
import argparse
import os
import io
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


_DEFAULT_BASE_DIR = os.path.expanduser('~/gensim-data')
BASE_DIR = os.environ.get('GENSIM_DATA_DIR', _DEFAULT_BASE_DIR)
"""The default location to store downloaded data.

You may override this with the GENSIM_DATA_DIR environment variable.

"""

_PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))


base_dir = BASE_DIR  # for backward compatibility with some of our test data

logger = logging.getLogger(__name__)

DATA_LIST_URL = "https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list.json"
DOWNLOAD_BASE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download"


def _progress(chunks_downloaded, chunk_size, total_size, part=1, total_parts=1):
    """Reporthook for :func:`urllib.urlretrieve`, code from [1]_.

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
    if not os.path.isdir(BASE_DIR):
        try:
            logger.info("Creating %s", BASE_DIR)
            os.makedirs(BASE_DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception(
                    "Not able to create folder gensim-data in {}. File gensim-data "
                    "exists in the directory already.".format(_PARENT_DIR)
                )
            else:
                raise Exception(
                    "Can't create {}. Make sure you have the read/write permissions "
                    "to the directory or you can try creating the folder manually"
                    .format(BASE_DIR)
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


def _load_info(url=DATA_LIST_URL, encoding='utf-8'):
    """Load dataset information from the network.

    If the network access fails, fall back to a local cache.  This cache gets
    updated each time a network request _succeeds_.
    """
    cache_path = os.path.join(BASE_DIR, 'information.json')
    _create_base_dir()

    try:
        info_bytes = urlopen(url).read()
    except (OSError, IOError):
        #
        # The exception raised by urlopen differs between Py2 and Py3.
        #
        # https://docs.python.org/3/library/urllib.error.html
        # https://docs.python.org/2/library/urllib.html
        #
        logger.exception(
            'caught non-fatal exception while trying to update gensim-data cache from %r; '
            'using local cache at %r instead', url, cache_path
        )
    else:
        with open(cache_path, 'wb') as fout:
            fout.write(info_bytes)

    try:
        #
        # We need io.open here because Py2 open doesn't support encoding keyword
        #
        with io.open(cache_path, 'r', encoding=encoding) as fin:
            return json.load(fin)
    except IOError:
        raise ValueError(
            'unable to read local cache %r during fallback, '
            'connect to the Internet and retry' % cache_path
        )


def info(name=None, show_only_latest=True, name_only=False):
    """Provide the information related to model/dataset.

    Parameters
    ----------
    name : str, optional
        Name of model/dataset.  If not set - shows all available data.
    show_only_latest : bool, optional
        If storage contains different versions for one data/model, this flag allow to hide outdated versions.
        Affects only if `name` is None.
    name_only : bool, optional
        If True, will return only the names of available models and corpora.

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
    .. sourcecode:: pycon

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
    information = _load_info()

    if name is not None:
        corpora = information['corpora']
        models = information['models']
        if name in corpora:
            return information['corpora'][name]
        elif name in models:
            return information['models'][name]
        else:
            raise ValueError("Incorrect model/corpus name")

    if not show_only_latest:
        return information

    if name_only:
        return {"corpora": list(information['corpora'].keys()), "models": list(information['models'])}

    return {
        "corpora": {name: data for (name, data) in information['corpora'].items() if data.get("latest", True)},
        "models": {name: data for (name, data) in information['models'].items() if data.get("latest", True)}
    }


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
    url_load_file = "{base}/{fname}/__init__.py".format(base=DOWNLOAD_BASE_URL, fname=name)
    data_folder_dir = os.path.join(BASE_DIR, name)
    data_folder_dir_tmp = data_folder_dir + '_tmp'
    tmp_dir = tempfile.mkdtemp()
    init_path = os.path.join(tmp_dir, "__init__.py")
    urllib.urlretrieve(url_load_file, init_path)
    total_parts = _get_parts(name)
    if total_parts > 1:
        concatenated_folder_name = "{fname}.gz".format(fname=name)
        concatenated_folder_dir = os.path.join(tmp_dir, concatenated_folder_name)
        for part in range(0, total_parts):
            url_data = "{base}/{fname}/{fname}.gz_0{part}".format(base=DOWNLOAD_BASE_URL, fname=name, part=part)

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
                part_path = os.path.join(tmp_dir, "{fname}.gz_0{part}".format(fname=name, part=part))
                with open(part_path, "rb") as rfp:
                    shutil.copyfileobj(rfp, wfp)
                os.remove(part_path)
    else:
        url_data = "{base}/{fname}/{fname}.gz".format(base=DOWNLOAD_BASE_URL, fname=name)
        fname = "{fname}.gz".format(fname=name)
        dst_path = os.path.join(tmp_dir, fname)
        urllib.urlretrieve(url_data, dst_path, reporthook=_progress)
        if _calculate_md5_checksum(dst_path) == _get_checksum(name):
            sys.stdout.write("\n")
            sys.stdout.flush()
            logger.info("%s downloaded", name)
        else:
            shutil.rmtree(tmp_dir)
            raise Exception("Checksum comparison failed, try again")

    if os.path.exists(data_folder_dir_tmp):
        os.remove(data_folder_dir_tmp)

    shutil.move(tmp_dir, data_folder_dir_tmp)
    os.rename(data_folder_dir_tmp, data_folder_dir)


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
    """Download (if needed) dataset/model and load it to memory (unless `return_path` is set).

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

    .. sourcecode:: pycon

        >>> import gensim.downloader as api
        >>>
        >>> model = api.load("glove-twitter-25")  # load glove vectors
        >>> model.most_similar("cat")  # show words that similar to word 'cat'

    Dataset example:

    .. sourcecode:: pycon

        >>> import gensim.downloader as api
        >>>
        >>> wiki = api.load("wiki-en")  # load extracted Wikipedia dump, around 6 Gb
        >>> for article in wiki:  # iterate over all wiki script
        >>>     pass

    Download only example:

    .. sourcecode:: pycon

        >>> import gensim.downloader as api
        >>>
        >>> print(api.load("wiki-en", return_path=True))  # output: /home/user/gensim-data/wiki-en/wiki-en.gz

    """
    _create_base_dir()
    file_name = _get_filename(name)
    if file_name is None:
        raise ValueError("Incorrect model/corpus name")
    folder_dir = os.path.join(BASE_DIR, name)
    path = os.path.join(folder_dir, file_name)
    if not os.path.exists(folder_dir):
        _download(name)

    if return_path:
        return path
    else:
        sys.path.insert(0, BASE_DIR)
        module = __import__(name)
        return module.load_data()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', stream=sys.stdout, level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description="Gensim console API",
        usage="python -m gensim.api.downloader  [-h] [-d data_name | -i data_name]"
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
        if args.info == 'name':
            print(json.dumps(info(name_only=True), indent=4))
        else:
            output = info() if (args.info == full_information) else info(name=args.info)
            print(json.dumps(output, indent=4))
