__author__ = 'lev'
import comparison
n_docs = 3

n_features = n_docs
n_topics = 3
n_jobs = n_docs
max_iterations = 50

batch_size = 3  # 1 for gs, 3 for skl
update_after = n_docs

kappa = 0.5  # batch 0.5  # decay in gensim
tau0 = 1.  # offest in gensim


def load_small_dataset():
    train = [u"apples", u"bananas", u"raspberries"]
    test = [u"bananas"]
    return train, test


def test_preprocessing():
    train, test = load_small_dataset()
    (train_X, test_X, vectorizer) = comparison.pre_processing(
        train_data=train, test_data=test, max_features=3, max_df=0.8, min_df=1)
    assert train_X.shape[0] == len(train)
    assert test_X.shape[0] == len(test)


def test_sklearn_batch_run():
    test_mode = 'batch'
    train_data, test_data = load_small_dataset()

    train_X, test_X, vectorizer = comparison.pre_processing(
        train_data=train_data, test_data=test_data, max_features=n_features, max_df=0.8, min_df=1)
    sklearn_perplexity = comparison.sklearn_run(test_mode=test_mode,
                                                train_X=train_X,
                                                test_X=test_X,
                                                n_topics=n_topics,
                                                n_jobs=1,
                                                max_iterations=max_iterations,
                                                vectorizer=vectorizer,
                                                offset=tau0,
                                                decay=kappa,
                                                total_samples=n_docs
                                                )

    assert sklearn_perplexity[0] == 11.182055702902652
    assert sklearn_perplexity[1] == 19.245542461891926
    # assert (sklearn_perplexity[2] == [[0.33492697, 1.3241937, 0.34087933]]).all
    # real result was 0.00591993331909

def test_gensim_preprocessing():
    train, test = load_small_dataset()
    (train_X, test_X, vectorizer) = comparison.pre_processing(
        train_data=train, test_data=test, max_features=3, max_df=0.8, min_df=1)

    train_corpus, test_corpus, id2words = comparison.gensim_prep(
        train_X=train_X, test_X=test_X, vectorizer=vectorizer)

    assert len(train_corpus) == len(train)
    assert list(train_corpus) == [[(0, 1)], [(1, 1)], [(2, 1)]]
    assert len(test_corpus) == len(test)
    assert list(test_corpus) == [[(1, 1)]]

def test_gensim_batch_run():
    test_mode = 'batch'

    train_data, test_data = load_small_dataset()

    train_X, test_X, vectorizer = comparison.pre_processing(
        train_data=train_data, test_data=test_data, max_features=n_features, max_df=0.8, min_df=1)

    train_corpus, test_corpus, id2words = comparison.gensim_prep(
        train_X=train_X, test_X=test_X, vectorizer=vectorizer)

    gensim_perplexity = comparison.gensim_run(test_mode=test_mode,
                                              train_corpus=train_corpus,
                                              test_corpus=test_corpus,
                                              n_topics=n_topics,
                                              n_jobs=n_jobs,
                                              max_iterations=max_iterations,
                                              id2words=id2words,
                                            batch_size=1,
                                              decay=kappa,
                                              offset=tau0,
                                              )

    print gensim_perplexity

    # should be this
    # assert gensim_perplexity[0] == 11.18202581586212
    # assert gensim_perplexity[1] == 19.259947213026244
    # in fact this
    assert gensim_perplexity[0] == 4.1740509613027834
    assert gensim_perplexity[1] == 1.0017670585302429


def test_gensim_singlecore_batch_run():
    test_mode = 'batch'

    train_data, test_data = load_small_dataset()

    train_X, test_X, vectorizer = comparison.pre_processing(
        train_data=train_data, test_data=test_data, max_features=n_features, max_df=0.8, min_df=1)

    train_corpus, test_corpus, id2words = comparison.gensim_prep(
        train_X=train_X, test_X=test_X, vectorizer=vectorizer)

    gensim_perplexity = comparison.gensim_single_core_run(test_mode=test_mode,
                                                          train_corpus=train_corpus,
                                                          test_corpus=test_corpus,
                                                          n_topics=n_topics,
                                                          n_jobs=n_jobs,
                                                          max_iterations=max_iterations,
                                                          id2words=id2words,
                                                          decay=kappa,
                                                          offset=tau0,
                                                          total_samples=n_docs,
                                                          batch_size=1
                                                          )

    print "This is gensim_perplexity"
    print gensim_perplexity
    assert gensim_perplexity[0] == 4.1740511571756409
    assert gensim_perplexity[1] == 1.0017672665292146
    # 11.207742578338911, 18.826747850838952
    # [[ 0.33800402  1.3144455   0.34755048]]


def test_sklearn_online_run():
    test_mode = 'online'
    train_data, test_data = load_small_dataset()

    train_X, test_X, vectorizer = comparison.pre_processing(
        train_data=train_data, test_data=test_data, max_features=n_features, max_df=0.8, min_df=1)
    sklearn_perplexity = comparison.sklearn_run(
        test_mode=test_mode,
        train_X=train_X,
        test_X=test_X,
        n_topics=n_topics,
        n_jobs=1,
        max_iterations=max_iterations,
        vectorizer=vectorizer,
        decay=kappa,
        offset=tau0,
        total_samples=n_docs,
        batch_size=batch_size)

    print sklearn_perplexity
    # expected this
    # assert sklearn_perplexity[0] == 10.476297933158504
    # assert sklearn_perplexity[1] == 21.579590404225367
    # got this results
    assert sklearn_perplexity[0] == 11.208575054734153
    assert sklearn_perplexity[1] == 18.825294617159187
    # assert (sklearn_perplexity[2] == [[0.33492697, 1.3241937, 0.34087933]]).all


def test_gensim_online_run():
    test_mode = 'online'
    train_data, test_data = load_small_dataset()

    train_X, test_X, vectorizer = comparison.pre_processing(
        train_data=train_data, test_data=test_data, max_features=n_features, max_df=0.8, min_df=1)

    train_corpus, test_corpus, id2words = comparison.gensim_prep(
        train_X=train_X, test_X=test_X, vectorizer=vectorizer)

    gensim_perplexity = comparison.gensim_run(test_mode=test_mode,
                                              train_corpus=train_corpus,
                                              test_corpus=test_corpus,
                                              n_topics=n_topics,
                                              n_jobs=n_jobs,
                                              max_iterations=max_iterations,
                                              id2words=id2words,
                                              decay=kappa,
                                              offset=tau0,
                                              total_samples=n_docs,
                                              batch_size=1, update_after=update_after  # one m-step per chunk

                                              )

    # this was expected
    # assert gensim_perplexity[0] == 12.074197413443118
    # assert gensim_perplexity[1] == 19.414047426272397
    # real test results
    from math import floor
    perplexity1 = floor(gensim_perplexity[0]*1000000)/1000000
    perplexity2 = floor(gensim_perplexity[1]*100000)/100000
    assert perplexity1 == 4.174050
    assert perplexity2 == 1.00176


def test_gensim_online_singlecore_run():
    test_mode = 'online'
    train_data, test_data = load_small_dataset()

    train_X, test_X, vectorizer = comparison.pre_processing(
        train_data=train_data, test_data=test_data, max_features=n_features, max_df=0.8, min_df=1)

    train_corpus, test_corpus, id2words = comparison.gensim_prep(
        train_X=train_X, test_X=test_X, vectorizer=vectorizer)

    gensim_perplexity = comparison.gensim_single_core_run(
        test_mode=test_mode,
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        n_topics=n_topics,
        n_jobs=n_jobs,
        max_iterations=max_iterations,
        id2words=id2words,
        decay=kappa,
        offset=tau0,
        total_samples=n_docs,
        batch_size=1)

    print gensim_perplexity

    # expected results
    # assert gensim_perplexity[0] == 10.474984476417523
    # assert gensim_perplexity[1] == 21.580473227971478
    # real test results
    from math import floor
    perplexity1 = floor(gensim_perplexity[0]*1000000)/1000000
    perplexity2 = floor(gensim_perplexity[1]*100000)/100000
    assert perplexity1 == 4.174051
    assert perplexity2 == 1.00176
