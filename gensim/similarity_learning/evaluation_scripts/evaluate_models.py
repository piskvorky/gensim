from gensim.similarity_learning import WikiQAExtractor
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import numpy as np
import pandas as pd
import argparse
import os
import logging

logger = logging.getLogger(__name__)

"""
This script should be run to get a full evaluation
"""


class LabeledLineSentence(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for uid, line in enumerate(self.corpus):
            yield TaggedDocument(words=line.split(), tags=['SENT_%s' % uid])


def mapk(Y_true, Y_pred):
    aps = []

    for y_true, y_pred in zip(Y_true, Y_pred):
        pred_sorted = sorted(zip(y_true, y_pred),
                             key=lambda x: x[1], reverse=True)
        avg = 0
        n_relevant = 0

        for i, val in enumerate(pred_sorted):
            if val[0] == 1:
                avg += 1. / (i + 1.)
                n_relevant += 1

        if n_relevant != 0:
            ap = avg / n_relevant
            aps.append(ap)
    return np.mean(np.array(aps))


def mean_ndcg(Y_true, Y_pred, k=10):
    ndcgs = []

    for y_true, y_pred in zip(Y_true, Y_pred):
        pred_sorted = sorted(zip(y_true, y_pred),
                             key=lambda x: x[1], reverse=True)
        true_sorted = sorted(zip(y_true, y_pred),
                             key=lambda x: x[0], reverse=True)

        pred_sorted = pred_sorted[:k]
        true_sorted = true_sorted[:k]

        dcg = 0
        for i, val in enumerate(pred_sorted):
            if val[0] == 1:
                dcg += 1. / np.log2(i + 2)

        idcg = 0
        for i, val in enumerate(true_sorted):
            if val[0] == 1:
                idcg += 1. / np.log2(i + 2)

        if idcg != 0:
            ndcgs.append(dcg / idcg)
    return np.mean(np.array(ndcgs))


def accuracy(Y_true, Y_pred):
    n_correct = 0
    for y_pred, y_true in zip(Y_pred, Y_true):
        if (np.argmax(y_true) == np.argmax(y_pred)):
            n_correct += 1

    return n_correct / len(Y_true)


def cos_sim(vec1, vec2):
    return np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def write_results_to_file(eval_metrics, file_to_write):
    """Writes the evaluated metrics in the given
    TODO
    """
    raise NotImplementedError()


def get_metric_results(Y_true, Y_pred, k_range=[1, 3, 5, 10, 20]):
    """returns a dict of calculated metrics
    """
    eval_metrics = {}
    eval_metrics["map"] = mapk(Y_true, Y_pred)

    for k in k_range:
        eval_metrics["ndcg@" + str(k)] = mean_ndcg(Y_true, Y_pred, k=k)

    eval_metrics["accuracy"] = accuracy(Y_true, Y_pred)

    return eval_metrics


def doc2vec_eval(datapath, vec_size=20, alpha=0.025, file_to_write=None):
    """Trains the doc2vec model on training data of WikiQA and then
    evaluates on test data
    """
    # load training data
    wikiqa_train = WikiQAExtractor(os.path.join(datapath, "WikiQA-train.tsv"))
    train_data = wikiqa_train.get_preprocessed_corpus()
    lls = LabeledLineSentence(train_data)

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm=1)
    model.build_vocab(lls)
    model.train(lls,
                total_examples=model.corpus_count,
                epochs=model.iter)

    # load test data
    wikiqa_test = WikiQAExtractor(os.path.join(datapath, "WikiQA-test.tsv"))
    test_doc_data = wikiqa_test.get_data()
    Y_true = []
    Y_pred = []

    for doc in test_doc_data:
        y_true = []
        y_pred = []
        for query, d, label in doc:
            y_pred.append(cos_sim(model.infer_vector(
                query), model.infer_vector(d)))
            y_true.append(label)
        Y_true.append(y_true)
        Y_pred.append(y_pred)
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    results = get_metric_results(Y_true, Y_pred)
    print("Results of evaluating on doc2vec:")
    print(results)

    if file_to_write is not None:
        write_results_to_file(results, file_to_write)

    # TODO might have to tune with validation data
    # TODO get params from user?


def word2vec_eval(datapath, word_embedding_path, file_to_write=None):
    """Averages words in a query to represent the sentence/doc
    If the word is out of vocabulary, we ignore it"""

    # load test data
    # Note: here we are not using train data to keep results consistent with
    # other models
    wikiqa_test = WikiQAExtractor(os.path.join(datapath, "WikiQA-test.tsv"))
    test_doc_data = wikiqa_test.get_data()

    # dict to store word-index pairs
    w2v = {}
    with open(word_embedding_path) as f:
        for line in f:
            string_array = np.array(line.split()[1:])
            string_array = [float(i) for i in string_array]
            w2v[line.split()[0]] = string_array

    def sent2vec(w2v, sentence):
        """Function to convert a sentence into an averaged vector
        """
        vec_sum = []
        for word in sentence.split():
            if word in w2v:
                vec_sum.append(w2v[word])
        return np.mean(np.array(vec_sum), axis=0)

    Y_true = []
    Y_pred = []
    for query_doc_group in test_doc_data:
        y_true = []
        y_pred = []
        for query, doc, label in query_doc_group:
            y_pred.append(cos_sim(sent2vec(w2v, query), sent2vec(w2v, doc)))
            y_true.append(label)
        Y_true.append(y_true)
        Y_pred.append(y_pred)
    Y_pred = np.array(Y_pred)
    Y_true = np.array(Y_true)

    results = get_metric_results(Y_true, Y_pred)

    print("Results of evaluating on word2vec:")
    print(results)

    if file_to_write is not None:
        write_results_to_file(results, file_to_write)
    # TODO we can do an evaluation on the whole dataset since this is unsupervised
    # currently WikiQAExtractor cannot support multiple files. Add it.
    # call it get_unsupervised_data or something

    # TODO add option to train on multiple w2v dimension files
    # Example: [50d, 100d, 200d, etc]

    # TODO maybe replace w2v dict which is memory intensice with gensim
    # KeyedVectors


def mz_eval(mz_output_file, file_to_write=None):
    with open(mz_output_file) as f:
        df = pd.read_csv(f, sep='\t')

    Y_true = []
    Y_pred = []

    for Question, Answer in df.groupby('QuestionID').apply(dict).items():

        this_y_true = []
        this_y_pred = []

        for d, l in zip(Answer['e'], Answer['g']):
            this_y_pred.append(d)
            this_y_true.append(l)

        Y_pred.append(this_y_pred)
        Y_true.append(this_y_true)

    results = get_metric_results(Y_true, Y_pred)

    print("Results of evaluating on mz_eval:")
    print(results)

    if file_to_write is not None:
        write_results_to_file(results, file_to_write)


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='all',
                        help='runs the evaluation of doc2vec')

    # Note: we currently only support WikiQA
    parser.add_argument('--datapath',
                        help='path to the folder with WikiQACorpus. Path should include WikiQACorpus\
                         Make sure you have run get_data.py in gensim/similarity_learning/data/')

    # TODO include gensim-data path to word embeddings
    parser.add_argument('--word_embedding_path',
                        help='path to the Glove word embedding file')

    parser.add_argument('--mz_output_file',
                        help='path to the prediction output file made by mz')

    args = parser.parse_args()
    if args.model == 'doc2vec':
        doc2vec_eval(args.datapath)
    elif args.model == 'word2vec':
        word2vec_eval(args.datapath, args.word_embedding_path)
    elif args.model == 'mz':
        mz_eval(args.mz_output_file)
    else:
        doc2vec_eval(args.datapath)
        word2vec_eval(args.datapath, args.word_embedding_path)
        mz_eval(args.mz_output_file)


"""
$ python evaluate_models.py --model doc2vec --datapath ../data/WikiQACorpus/
$ python evaluate_models.py --model word2vec --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt  # noqa:F401
"""
