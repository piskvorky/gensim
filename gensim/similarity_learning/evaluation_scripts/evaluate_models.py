from gensim.similarity_learning import WikiQAExtractor
from gensim.similarity_learning import WikiQA_DRMM_TKS_Extractor
from gensim.similarity_learning.preprocessing import ListGenerator
from gensim.similarity_learning.models import DRMM_TKS
from gensim.similarity_learning import rank_hinge_loss
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.similarity_learning import ValidationCallback
from gensim.similarity_learning import mapk, mean_ndcg
import numpy as np
import pandas as pd
import argparse
import os
import logging
import time

logger = logging.getLogger(__name__)

"""
This script should be run to get a model by model based or full evaluation
Make sure you run gensim/similarity_learning/data/get_data.py to get the datasets

Currently supports
- DRMM TKS
- doc2vec
- word2vec
- MatchZoo models

Example usage:
==============

For evaluating drmm_tks on the WikiQA corpus
$ python evaluate_models.py --model drmm_tks --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt --result_save_path results_drmm_tks

For evaluating doc2vec on the WikiQA corpus
$ python evaluate_models.py --model doc2vec --datapath ../data/WikiQACorpus/ --result_save_path results_d2v

For evaluating word2vec averaging on the WikiQA corpus
$ python evaluate_models.py --model word2vec --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt  --result_save_path results_w2v# noqa:F401

For evaluating the TREC format file produced by MatchZoo:
$ python evaluate_models.py  --model mz --mz_result_file mz_results/predict.test.anmm.wikiqa.txt
Note: here "predict.test.anmm.wikiqa.txt" is the file output by MZ. It has been provided in this repo as an example in mz_eval/

For evaluating all models
-with one mz output file
$ python evaluate_models.py --model all --mz_result_file mz_results/predict.test.anmm.wikiqa.txt --result_save_path results_mz_file --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt --datapath ../data/WikiQACorpus/  # noqa:F401
-with a mz folder filled with result files
$ python evaluate_models.py  --model all --mz_result_folder mz_results/ --result_save_path results_all --datapath ../data/WikiQACorpus/ --word_embedding_path ../evaluation_scripts/glove.6B.50d.txt  # noqa:F401
"""


class LabeledLineSentence(object):
    """class to make sentences iterable
    """

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for uid, line in enumerate(self.corpus):
            yield TaggedDocument(words=line.split(), tags=['SENT_%s' % uid])


# list to store results from all models to be saved later
results_list = []


def cos_sim(vec1, vec2):
    """Calculates the cosine similarity of 2 vectos
    """
    return np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_metric_results(Y_true, Y_pred, k_range=[1, 3, 5, 10, 20]):
    """returns a dict of calculated metrics
    """
    eval_metrics = {}
    eval_metrics["map"] = mapk(Y_true, Y_pred)

    for k in k_range:
        eval_metrics["ndcg@" + str(k)] = mean_ndcg(Y_true, Y_pred, k=k)

    return eval_metrics


def doc2vec_eval(datapath, vec_size=20, alpha=0.025):
    """Trains the doc2vec model on training data of WikiQA and then
    evaluates on test data

    parameters:
    ==========

    datapath : string
        path to the WikiQA folder

    vec_size : int
        size of the hidden layer

    alpha : float
        The initial learning rate.
    """
    # load testing data
    wikiqa_train = WikiQAExtractor(os.path.join(datapath, "WikiQA-test.tsv"))
    train_data = wikiqa_train.get_preprocessed_corpus()
    lls = LabeledLineSentence(train_data)

    initial_time = time.time()
    logger.info("Building and training doc2vec model")
    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm=1,
                    iter=50)
    model.build_vocab(lls)
    model.train(lls,
                total_examples=model.corpus_count,
                epochs=model.iter)
    logger.info("Building and training of doc2vec done")
    logger.info("Time taken to train is %f" %
                float(time.time() - initial_time))

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
    results["method"] = "d2v"
    logger.info("Results of evaluating on doc2vec:")
    logger.info(results)

    results_list.append(results)

    # TODO might have to tune with validation data
    # TODO get params from user?


def word2vec_eval(datapath, word_embedding_path):
    """Averages words in a query to represent the sentence/doc
    If the word is out of vocabulary, we ignore it

    parameters:
    ==========
    datapath : string
        path to the WikiQA folder

    word_embedding_path : string
        path to the the .txt which has the Glove word embeddings
    """
    # load test data
    # Note: here we are not using train data to keep results consistent with
    # other models
    wikiqa_test = WikiQAExtractor(os.path.join(datapath, "WikiQA-test.tsv"))
    test_doc_data = wikiqa_test.get_data()

    logger.info("Starting building word-vec dict")
    # dict to store word-index pairs
    w2v = {}
    with open(word_embedding_path) as f:
        for line in f:
            string_array = np.array(line.split()[1:])
            string_array = [float(i) for i in string_array]
            w2v[line.split()[0]] = string_array
    logger.info("Word-vec dict build complete")

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
    results["method"] = "w2v"

    logger.info("Results of evaluating on word2vec:")
    logger.info(results)

    results_list.append(results)
    # TODO we can do an evaluation on the whole dataset since this is unsupervised
    # currently WikiQAExtractor cannot support multiple files. Add it.
    # call it get_unsupervised_data or something

    # TODO add option to train on multiple w2v dimension files
    # Example: [50d, 100d, 200d, etc]

    # TODO maybe replace w2v dict which is memory intensice with gensim
    # KeyedVectors


def mz_eval(mz_output_file):
    """Evaluates the metrics on a TREC format file output by MatchZoo

    parameters:
    ==========
    mz_output_file : string
        path to MatchZoo output TREC format file
    """

    with open(mz_output_file) as f:
        df = pd.read_csv(f, sep='\t', names=[
            "QuestionID", "Q0", "Doc ID", "Doc No", "predicted_score", "model name", "actual_score"])

    Y_true = []
    Y_pred = []

    # Group the results based on QuestionID column
    for Question, Answer in df.groupby('QuestionID').apply(dict).items():
        y_true = []
        y_pred = []

        for d, l in zip(Answer['predicted_score'], Answer['actual_score']):
            y_pred.append(d)
            y_true.append(l)

        Y_pred.append(y_pred)
        Y_true.append(y_true)

    results = get_metric_results(Y_true, Y_pred)
    results["method"] = "MZ"  # TODO add a way to specify the function

    logger.info("Results of evaluating on mz_eval:")
    logger.info(results)

    results_list.append(results)


def mz_eval_multiple(mz_output_file_dir):
    """Evaluates multiple TREC format file output by MatchZoo

    parameters:
    ==========
    mz_output_file_dir : string
        path to folder with MatchZoo output TREC format files
    """

    for mz_output_file in os.listdir(mz_output_file_dir):
        with open(os.path.join(mz_output_file_dir, mz_output_file)) as f:
            df = pd.read_csv(f, sep='\t', names=[
                             "QuestionID", "Q0", "Doc ID", "Doc No", "predicted_score", "model name", "actual_score"])

        Y_true = []
        Y_pred = []

        # Group the results based on QuestionID column
        for Question, Answer in df.groupby('QuestionID').apply(dict).items():
            y_true = []
            y_pred = []

            for d, l in zip(Answer['predicted_score'], Answer['actual_score']):
                y_pred.append(d)
                y_true.append(l)

            Y_pred.append(y_pred)
            Y_true.append(y_true)

        results = get_metric_results(Y_true, Y_pred)
        results["method"] = mz_output_file.split('.')[2]

        logger.info("Results of evaluating on mz_eval:")
        logger.info(results)

        results_list.append(results)


def drmm_tks_eval(datapath, word_embedding_path):
    # Here, we'll add the test file. You can also use the validation file if
    # needed.
    train_file_path = os.path.join(datapath, 'WikiQA-train.tsv')
    test_file_path = os.path.join(datapath, 'WikiQA-test.tsv')
    dev_file_path = os.path.join(datapath, 'WikiQA-dev.tsv')
    word_embedding_path = word_embedding_path

    wikiqa_train = WikiQA_DRMM_TKS_Extractor(file_path=train_file_path, word_embedding_path=word_embedding_path,
                                             keep_full_embedding=True, text_maxlen=140)

    dev_list_gen = ListGenerator(dev_file_path, text_maxlen=wikiqa_train.text_maxlen,
                                 train_word2index=wikiqa_train.word2index, additional_word2index=wikiqa_train.additional_word2index,
                                 oov_handle_method="ignore", zero_word_index=wikiqa_train.zero_word_index, train_pad_word_index=wikiqa_train.pad_word_index)

    test_list_gen = ListGenerator(test_file_path, text_maxlen=wikiqa_train.text_maxlen,
                                  train_word2index=wikiqa_train.word2index, additional_word2index=wikiqa_train.additional_word2index,
                                  oov_handle_method="ignore", zero_word_index=wikiqa_train.zero_word_index, train_pad_word_index=wikiqa_train.pad_word_index)

    X1_train, X2_train, y_train = wikiqa_train.get_full_batch()
    drmm_tks = DRMM_TKS(embedding=wikiqa_train.embedding_matrix, vocab_size=wikiqa_train.embedding_matrix.shape[0],
                        text_maxlen=wikiqa_train.text_maxlen)

    model = drmm_tks.get_model()
    model.summary()

    optimizer = 'adadelta'
    loss = 'mse'

    validation_data = dev_list_gen.get_list_data()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(x={"query": X1_train, "doc": X2_train}, y=y_train, batch_size=64,
              verbose=1, epochs=10, shuffle=True, callbacks=[ValidationCallback(validation_data)])

    data = test_list_gen.get_list_data()
    X1 = data["X1"]
    X2 = data["X2"]
    y = data["y"]
    doc_lengths = data["doc_lengths"]

    predictions = model.predict(x={"query": X1, "doc": X2})

    Y_pred = []
    Y_true = []
    offset = 0

    for doc_size in doc_lengths:
        Y_pred.append(predictions[offset: offset + doc_size])
        Y_true.append(y[offset: offset + doc_size])
        offset += doc_size

    results = get_metric_results(Y_true, Y_pred)
    results["method"] = "DRMM_TKS"  # TODO add a way to specify the function

    logger.info("Results of evaluating on DRMM_TKS:")
    logger.info(results)

    results_list.append(results)


def write_results_to_file(results_list, file_to_write, k_range=[1, 3, 5, 10, 20]):
    """Writes the evaluated metrics in the given
    """
    k_range = ["ndcg@" + str(k) for k in k_range]
    metric_order = ["map"] + k_range

    header = "Method, "
    for metric in metric_order:
        header += metric + ", "
    header += "\n"

    to_write = ""
    for result in results_list:
        to_write += result["method"] + ", "
        for metric in metric_order:
            to_write += str(result[metric]) + ", "
        to_write += "\n"

    with open(file_to_write + ".csv", "w") as f:
        f.write(header)
        f.write(to_write)

    logger.info("Results saved in %s" % file_to_write + ".csv")


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='all',
                        help='runs the evaluation on the given model type. Options are:\
                        doc2vec, word2vec, mz_eval, mz_eval_multiple')

    # Note: we currently only support WikiQA
    parser.add_argument('--datapath',
                        help='path to the folder with WikiQACorpus. Path should include WikiQACorpus\
                         Make sure you have run get_data.py in gensim/similarity_learning/data/')

    # TODO include gensim-data path to word embeddings
    parser.add_argument('--word_embedding_path',
                        help='path to the Glove word embedding file')

    parser.add_argument('--mz_result_file',
                        help='path to the prediction output file made by mz')

    parser.add_argument('--result_save_path',
                        default=None,
                        help='path to save the results to as a csv')

    parser.add_argument('--mz_result_folder',
                        default=None,
                        help='path to mz folder with many test prediction outputs')

    args = parser.parse_args()
    if args.model == 'doc2vec':
        doc2vec_eval(args.datapath)
    elif args.model == 'word2vec':
        word2vec_eval(args.datapath, args.word_embedding_path)
    elif args.model == 'mz':
        mz_eval(args.mz_result_file)
    elif args.model == 'mz_folder':
        mz_eval_multiple(args.mz_result_folder)
    elif args.model == 'drmm_tks':
        if not (args.datapath or args.word_embedding_path):
            parser.error(
                "You need to specify --datapath and --word_embedding_path")
        drmm_tks_eval(args.datapath, args.word_embedding_path)
    elif args.model == 'all':
        doc2vec_eval(args.datapath)
        word2vec_eval(args.datapath, args.word_embedding_path)
        if args.mz_result_file is not None:
            mz_eval(args.mz_result_file)
        elif args.mz_result_folder is not None:
            mz_eval_multiple(args.mz_result_folder)

    if (args.result_save_path or results_list) is not None:
        write_results_to_file(results_list, args.result_save_path)
