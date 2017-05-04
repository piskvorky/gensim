from random import sample
import numpy as np
from gensim.matutils import kullback_leibler, hellinger, jaccard_set
from gensim.models.ldamulticore import LdaMulticore


def topic2topic_difference(m1, m2, distance="kulback_leibler", num_words=100, n_ann_terms=10):
    """
    Calculate difference topic2topic between two `LdaMulticore` models

    `m1` and `m2` are trained instances of `LdaMulticore`
    `distance` is function that will be applied to calculate difference between any topic pair.
    Available values: `kulback_leibler`, `hellinger` and `jaccard`
    `num_words` is quantity of most relevant words that used if distance == `jaccard` (also used for annotation)
    `n_ann_terms` is max quantity of words in intersection/symmetric difference between topics (used for annotation)

    Returns a matrix Z with shape (m1.num_topics, m2.num_topics), where Z[i][j] - difference between topic_i and topic_j
    and matrix annotation with shape (m1.num_topics, m2.num_topics, 2, None),
    where
        annotation[i][j] = [[`int_1`, `int_2`, ...], [`diff_1`, `diff_2`, ...]] and
        `int_k` is word from intersection of `topic_i` and `topic_j` and
        `diff_l` is word from symmetric difference of `topic_i` and `topic_j`

    Example:

    >>> m1, m2 = LdaMulticore.load(path_1), LdaMulticore.load(path_2)
    >>> mdiff, annotation = topic2topic_difference(m1, m2)
    >>> print(mdiff) # get matrix with difference for each topic pair from `m1` and `m2`
    >>> print(annotation) # get array with positive/negative words for each topic pair from `m1` and `m2`
    """

    distances = {"kulback_leibler": kullback_leibler,
                 "hellinger": hellinger,
                 "jaccard": jaccard_set}

    assert distance in distances, "Incorrect distance, valid only {}".format(", ".join("`{}`".format(x)
                                                                                       for x in distances.keys()))
    assert isinstance(m1, LdaMulticore), "The parameter `m1` must be of type `{}`".format(LdaMulticore.__name__)
    assert isinstance(m2, LdaMulticore), "The parameter `m2` must be of type `{}`".format(LdaMulticore.__name__)

    distance_func = distances[distance]
    d1, d2 = m1.state.get_lambda(), m2.state.get_lambda()
    t1_size, t2_size = d1.shape[0], d2.shape[0]

    fst_topics, snd_topics = None, None

    if distance == "jaccard":
        d1 = fst_topics = [{w for (w, _) in m1.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        d2 = snd_topics = [{w for (w, _) in m2.show_topic(topic, topn=num_words)} for topic in range(t2_size)]

    z = np.zeros((t1_size, t2_size))

    for topic1 in range(t1_size):
        for topic2 in range(t2_size):
            if topic2 < topic1:
                continue

            z[topic1][topic2] = z[topic2][topic1] = distance_func(d1[topic1], d2[topic2])

    z /= np.max(z)
    annotation = [[None for _ in range(t1_size)] for _ in range(t2_size)]

    if fst_topics is None or snd_topics is None:
        fst_topics = [{w for (w, _) in m1.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        snd_topics = [{w for (w, _) in m2.show_topic(topic, topn=num_words)} for topic in range(t2_size)]

    for topic1 in range(t1_size):
        for topic2 in range(t2_size):
            if topic2 < topic1:
                continue

            pos_tokens = fst_topics[topic1] & snd_topics[topic2]
            neg_tokens = fst_topics[topic1].symmetric_difference(snd_topics[topic2])

            pos_tokens = sample(pos_tokens, min(len(pos_tokens), n_ann_terms))
            neg_tokens = sample(neg_tokens, min(len(neg_tokens), n_ann_terms))

            annotation[topic1][topic2] = annotation[topic2][topic1] = [pos_tokens, neg_tokens]

    return z, annotation
