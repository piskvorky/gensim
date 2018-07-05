import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


def mapk(Y_true, Y_pred):
    """Calculates Mean Average Precision(MAP) for a given set of Y_true, Y_pred

    Note: Currently doesn't support mapping at k. Couldn't use only map as it's a
    reserved word

    Parameters
    ----------
    Y_true : numpy array or list of ints either 1 or 0
        Contains the true, ground truth values of the relevance between a query and document
    Y_pred : numpy array or list of floats
        Contains the predicted similarity score between a query and document

    Examples
    --------
    >>> Y_true = [[0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]]
    >>> Y_pred = [[0.1, 0.2, -0.01, 0.4], [0.12, -0.43, 0.2, 0.1, 0.99, 0.7], [0.5, 0.63, 0.92]]
    >>> print(mapk(Y_true, Y_pred))
    0.75
    """
    aps = []
    n_skipped = 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        # skip datapoints where there is no solution
        if np.sum(y_true) < 1:
            n_skipped += 1
            continue

        pred_sorted = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
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
    """Calculates the mean discounted normalized cumulative gain over all
    the entries limited to the integer k

    Parameters
    ----------
    Y_true : numpy array or list of ints either 1 or 0
        Contains the true, ground truth values of the relevance between a query and document
    Y_pred : numpy array or list of floats
        Contains the predicted similarity score between a query and document


    Examples
    --------
    >>> Y_true = [[0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0]]
    >>> Y_pred = [[0.1, 0.2, -0.01, 0.4], [0.12, -0.43, 0.2, 0.1, 0.19, 0.7], [0.5, 0.63, 0.72]]
    >>> for k in [1, 3, 5, 10]:
    ...     print("nDCG@{} is {}".format(k, mean_ndcg(Y_true, Y_pred, k)))
    nDCG@1 is 0.3333333333333333
    nDCG@3 is 0.7103099178571526
    nDCG@5 is 0.7103099178571526
    nDCG@10 is 0.7103099178571526

    """
    ndcgs = []
    n_skipped = 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        if np.sum(y_true) < 1:
            n_skipped += 1
            continue
        pred_sorted = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
        true_sorted = sorted(zip(y_true, y_pred), key=lambda x: x[0], reverse=True)
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
