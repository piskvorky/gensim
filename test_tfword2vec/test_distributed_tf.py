from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gensim
import tensorflow as tf
import os
import sys
import time
import argparse
import zipfile
import urllib

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def main(_):
    filename = maybe_download('text8.zip', 31344016)
    vocabulary = read_data(filename)
    tfw2v = gensim.models.TfWord2Vec(vocabulary, train_epochs=3, batch_size=1000,
                                     num_skips=2, window=1, size=128, negative=64,
                                     alpha=1, eval_data='questions-words.txt', FLAGS=FLAGS)
    tfw2v.build_dataset()
    start = time.time()
    tfw2v.train()
    print("Time:", time.time() - start)
    if tfw2v.start_index == 0:
        tfw2v.save("/models/tfw2v_model")
        tfw2v.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument("--ps_hosts", type=str, default="",
                            help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default="",
                            help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name", type=str, default="",
                                                help="One of 'ps', 'worker'")
    # Flags for defining the tf.train.Server
    parser.add_argument("--task_index", type=int, default=0,
                                            help="Index of task within the job")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
