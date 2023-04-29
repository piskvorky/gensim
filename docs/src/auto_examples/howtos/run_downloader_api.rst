.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_howtos_run_downloader_api.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_howtos_run_downloader_api.py:


How to download pre-trained models and corpora
==============================================

Demonstrates simple and quick access to common corpora and pretrained models.


.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)








One of Gensim's features is simple and easy access to common data.
The `gensim-data <https://github.com/RaRe-Technologies/gensim-data>`_ project stores a
variety of corpora and pretrained models.
Gensim has a :py:mod:`gensim.downloader` module for programmatically accessing this data.
This module leverages a local cache (in user's home folder, by default) that
ensures data is downloaded at most once.

This tutorial:

* Downloads the text8 corpus, unless it is already on your local machine
* Trains a Word2Vec model from the corpus (see :ref:`sphx_glr_auto_examples_tutorials_run_doc2vec_lee.py` for a detailed tutorial)
* Leverages the model to calculate word similarity
* Demonstrates using the API to load other models and corpora

Let's start by importing the api module.



.. code-block:: default

    import gensim.downloader as api








Now, let's download the text8 corpus and load it as a Python object
that supports streamed access.



.. code-block:: default

    corpus = api.load('text8')








In this case, our corpus is an iterable.
If you look under the covers, it has the following definition:


.. code-block:: default


    import inspect
    print(inspect.getsource(corpus.__class__))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    class Dataset(object):
        def __init__(self, fn):
            self.fn = fn

        def __iter__(self):
            corpus = Text8Corpus(self.fn)
            for doc in corpus:
                yield doc





For more details, look inside the file that defines the Dataset class for your particular resource.



.. code-block:: default

    print(inspect.getfile(corpus.__class__))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/kofola3/gensim-data/text8/__init__.py




With the corpus has been downloaded and loaded, let's use it to train a word2vec model.



.. code-block:: default


    from gensim.models.word2vec import Word2Vec
    model = Word2Vec(corpus)








Now that we have our word2vec model, let's find words that are similar to 'tree'.



.. code-block:: default



    print(model.wv.most_similar('tree'))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [('trees', 0.7091131806373596), ('bark', 0.673214316368103), ('leaf', 0.6706242561340332), ('flower', 0.6195512413978577), ('bird', 0.6081331372261047), ('nest', 0.602649450302124), ('avl', 0.5914573669433594), ('garden', 0.5712863206863403), ('egg', 0.5702848434448242), ('beetle', 0.5701731443405151)]




You can use the API to download several different corpora and pretrained models.
Here's how to list all resources available in gensim-data:



.. code-block:: default



    import json
    info = api.info()
    print(json.dumps(info, indent=4))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {
        "corpora": {
            "semeval-2016-2017-task3-subtaskBC": {
                "num_records": -1,
                "record_format": "dict",
                "file_size": 6344358,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/semeval-2016-2017-task3-subtaskB-eng/__init__.py",
                "license": "All files released for the task are free for general research use",
                "fields": {
                    "2016-train": [
                        "..."
                    ],
                    "2016-dev": [
                        "..."
                    ],
                    "2017-test": [
                        "..."
                    ],
                    "2016-test": [
                        "..."
                    ]
                },
                "description": "SemEval 2016 / 2017 Task 3 Subtask B and C datasets contain train+development (317 original questions, 3,169 related questions, and 31,690 comments), and test datasets in English. The description of the tasks and the collected data is given in sections 3 and 4.1 of the task paper http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-report.pdf linked in section \u201cPapers\u201d of https://github.com/RaRe-Technologies/gensim-data/issues/18.",
                "checksum": "701ea67acd82e75f95e1d8e62fb0ad29",
                "file_name": "semeval-2016-2017-task3-subtaskBC.gz",
                "read_more": [
                    "http://alt.qcri.org/semeval2017/task3/",
                    "http://alt.qcri.org/semeval2017/task3/data/uploads/semeval2017-task3.pdf",
                    "https://github.com/RaRe-Technologies/gensim-data/issues/18",
                    "https://github.com/Witiko/semeval-2016_2017-task3-subtaskB-english"
                ],
                "parts": 1
            },
            "semeval-2016-2017-task3-subtaskA-unannotated": {
                "num_records": 189941,
                "record_format": "dict",
                "file_size": 234373151,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/semeval-2016-2017-task3-subtaskA-unannotated-eng/__init__.py",
                "license": "These datasets are free for general research use.",
                "fields": {
                    "THREAD_SEQUENCE": "",
                    "RelQuestion": {
                        "RELQ_CATEGORY": "question category, according to the Qatar Living taxonomy",
                        "RELQ_DATE": "date of posting",
                        "RELQ_ID": "question indentifier",
                        "RELQ_USERID": "identifier of the user asking the question",
                        "RELQ_USERNAME": "name of the user asking the question",
                        "RelQBody": "body of question",
                        "RelQSubject": "subject of question"
                    },
                    "RelComments": [
                        {
                            "RelCText": "text of answer",
                            "RELC_USERID": "identifier of the user posting the comment",
                            "RELC_ID": "comment identifier",
                            "RELC_USERNAME": "name of the user posting the comment",
                            "RELC_DATE": "date of posting"
                        }
                    ]
                },
                "description": "SemEval 2016 / 2017 Task 3 Subtask A unannotated dataset contains 189,941 questions and 1,894,456 comments in English collected from the Community Question Answering (CQA) web forum of Qatar Living. These can be used as a corpus for language modelling.",
                "checksum": "2de0e2f2c4f91c66ae4fcf58d50ba816",
                "file_name": "semeval-2016-2017-task3-subtaskA-unannotated.gz",
                "read_more": [
                    "http://alt.qcri.org/semeval2016/task3/",
                    "http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-report.pdf",
                    "https://github.com/RaRe-Technologies/gensim-data/issues/18",
                    "https://github.com/Witiko/semeval-2016_2017-task3-subtaskA-unannotated-english"
                ],
                "parts": 1
            },
            "patent-2017": {
                "num_records": 353197,
                "record_format": "dict",
                "file_size": 3087262469,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/patent-2017/__init__.py",
                "license": "not found",
                "description": "Patent Grant Full Text. Contains the full text including tables, sequence data and 'in-line' mathematical expressions of each patent grant issued in 2017.",
                "checksum-0": "818501f0b9af62d3b88294d86d509f8f",
                "checksum-1": "66c05635c1d3c7a19b4a335829d09ffa",
                "file_name": "patent-2017.gz",
                "read_more": [
                    "http://patents.reedtech.com/pgrbft.php"
                ],
                "parts": 2
            },
            "quora-duplicate-questions": {
                "num_records": 404290,
                "record_format": "dict",
                "file_size": 21684784,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/quora-duplicate-questions/__init__.py",
                "license": "probably https://www.quora.com/about/tos",
                "fields": {
                    "question1": "the full text of each question",
                    "question2": "the full text of each question",
                    "qid1": "unique ids of each question",
                    "qid2": "unique ids of each question",
                    "id": "the id of a training set question pair",
                    "is_duplicate": "the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise"
                },
                "description": "Over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line contains a duplicate pair or not.",
                "checksum": "d7cfa7fbc6e2ec71ab74c495586c6365",
                "file_name": "quora-duplicate-questions.gz",
                "read_more": [
                    "https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs"
                ],
                "parts": 1
            },
            "wiki-english-20171001": {
                "num_records": 4924894,
                "record_format": "dict",
                "file_size": 6516051717,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/wiki-english-20171001/__init__.py",
                "license": "https://dumps.wikimedia.org/legal.html",
                "fields": {
                    "section_texts": "list of body of sections",
                    "section_titles": "list of titles of sections",
                    "title": "Title of wiki article"
                },
                "description": "Extracted Wikipedia dump from October 2017. Produced by `python -m gensim.scripts.segment_wiki -f enwiki-20171001-pages-articles.xml.bz2 -o wiki-en.gz`",
                "checksum-0": "a7d7d7fd41ea7e2d7fa32ec1bb640d71",
                "checksum-1": "b2683e3356ffbca3b6c2dca6e9801f9f",
                "checksum-2": "c5cde2a9ae77b3c4ebce804f6df542c2",
                "checksum-3": "00b71144ed5e3aeeb885de84f7452b81",
                "file_name": "wiki-english-20171001.gz",
                "read_more": [
                    "https://dumps.wikimedia.org/enwiki/20171001/"
                ],
                "parts": 4
            },
            "text8": {
                "num_records": 1701,
                "record_format": "list of str (tokens)",
                "file_size": 33182058,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/__init__.py",
                "license": "not found",
                "description": "First 100,000,000 bytes of plain text from Wikipedia. Used for testing purposes; see wiki-english-* for proper full Wikipedia datasets.",
                "checksum": "68799af40b6bda07dfa47a32612e5364",
                "file_name": "text8.gz",
                "read_more": [
                    "https://mattmahoney.net/dc/textdata.html"
                ],
                "parts": 1
            },
            "fake-news": {
                "num_records": 12999,
                "record_format": "dict",
                "file_size": 20102776,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/fake-news/__init__.py",
                "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                "fields": {
                    "crawled": "date the story was archived",
                    "ord_in_thread": "",
                    "published": "date published",
                    "participants_count": "number of participants",
                    "shares": "number of Facebook shares",
                    "replies_count": "number of replies",
                    "main_img_url": "image from story",
                    "spam_score": "data from webhose.io",
                    "uuid": "unique identifier",
                    "language": "data from webhose.io",
                    "title": "title of story",
                    "country": "data from webhose.io",
                    "domain_rank": "data from webhose.io",
                    "author": "author of story",
                    "comments": "number of Facebook comments",
                    "site_url": "site URL from BS detector",
                    "text": "text of story",
                    "thread_title": "",
                    "type": "type of website (label from BS detector)",
                    "likes": "number of Facebook likes"
                },
                "description": "News dataset, contains text and metadata from 244 websites and represents 12,999 posts in total from a specific window of 30 days. The data was pulled using the webhose.io API, and because it's coming from their crawler, not all websites identified by their BS Detector are present in this dataset. Data sources that were missing a label were simply assigned a label of 'bs'. There are (ostensibly) no genuine, reliable, or trustworthy news sources represented in this dataset (so far), so don't trust anything you read.",
                "checksum": "5e64e942df13219465927f92dcefd5fe",
                "file_name": "fake-news.gz",
                "read_more": [
                    "https://www.kaggle.com/mrisdal/fake-news"
                ],
                "parts": 1
            },
            "20-newsgroups": {
                "num_records": 18846,
                "record_format": "dict",
                "file_size": 14483581,
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/20-newsgroups/__init__.py",
                "license": "not found",
                "fields": {
                    "topic": "name of topic (20 variant of possible values)",
                    "set": "marker of original split (possible values 'train' and 'test')",
                    "data": "",
                    "id": "original id inferred from folder name"
                },
                "description": "The notorious collection of approximately 20,000 newsgroup posts, partitioned (nearly) evenly across 20 different newsgroups.",
                "checksum": "c92fd4f6640a86d5ba89eaad818a9891",
                "file_name": "20-newsgroups.gz",
                "read_more": [
                    "http://qwone.com/~jason/20Newsgroups/"
                ],
                "parts": 1
            },
            "__testing_matrix-synopsis": {
                "description": "[THIS IS ONLY FOR TESTING] Synopsis of the movie matrix.",
                "checksum": "1767ac93a089b43899d54944b07d9dc5",
                "file_name": "__testing_matrix-synopsis.gz",
                "read_more": [
                    "http://www.imdb.com/title/tt0133093/plotsummary?ref_=ttpl_pl_syn#synopsis"
                ],
                "parts": 1
            },
            "__testing_multipart-matrix-synopsis": {
                "description": "[THIS IS ONLY FOR TESTING] Synopsis of the movie matrix.",
                "checksum-0": "c8b0c7d8cf562b1b632c262a173ac338",
                "checksum-1": "5ff7fc6818e9a5d9bc1cf12c35ed8b96",
                "checksum-2": "966db9d274d125beaac7987202076cba",
                "file_name": "__testing_multipart-matrix-synopsis.gz",
                "read_more": [
                    "http://www.imdb.com/title/tt0133093/plotsummary?ref_=ttpl_pl_syn#synopsis"
                ],
                "parts": 3
            }
        },
        "models": {
            "fasttext-wiki-news-subwords-300": {
                "num_records": 999999,
                "file_size": 1005007116,
                "base_dataset": "Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/fasttext-wiki-news-subwords-300/__init__.py",
                "license": "https://creativecommons.org/licenses/by-sa/3.0/",
                "parameters": {
                    "dimension": 300
                },
                "description": "1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).",
                "read_more": [
                    "https://fasttext.cc/docs/en/english-vectors.html",
                    "https://arxiv.org/abs/1712.09405",
                    "https://arxiv.org/abs/1607.01759"
                ],
                "checksum": "de2bb3a20c46ce65c9c131e1ad9a77af",
                "file_name": "fasttext-wiki-news-subwords-300.gz",
                "parts": 1
            },
            "conceptnet-numberbatch-17-06-300": {
                "num_records": 1917247,
                "file_size": 1225497562,
                "base_dataset": "ConceptNet, word2vec, GloVe, and OpenSubtitles 2016",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/conceptnet-numberbatch-17-06-300/__init__.py",
                "license": "https://github.com/commonsense/conceptnet-numberbatch/blob/master/LICENSE.txt",
                "parameters": {
                    "dimension": 300
                },
                "description": "ConceptNet Numberbatch consists of state-of-the-art semantic vectors (also known as word embeddings) that can be used directly as a representation of word meanings or as a starting point for further machine learning. ConceptNet Numberbatch is part of the ConceptNet open data project. ConceptNet provides lots of ways to compute with word meanings, one of which is word embeddings. ConceptNet Numberbatch is a snapshot of just the word embeddings. It is built using an ensemble that combines data from ConceptNet, word2vec, GloVe, and OpenSubtitles 2016, using a variation on retrofitting.",
                "read_more": [
                    "http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14972",
                    "https://github.com/commonsense/conceptnet-numberbatch",
                    "http://conceptnet.io/"
                ],
                "checksum": "fd642d457adcd0ea94da0cd21b150847",
                "file_name": "conceptnet-numberbatch-17-06-300.gz",
                "parts": 1
            },
            "word2vec-ruscorpora-300": {
                "num_records": 184973,
                "file_size": 208427381,
                "base_dataset": "Russian National Corpus (about 250M words)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-ruscorpora-300/__init__.py",
                "license": "https://creativecommons.org/licenses/by/4.0/deed.en",
                "parameters": {
                    "dimension": 300,
                    "window_size": 10
                },
                "description": "Word2vec Continuous Skipgram vectors trained on full Russian National Corpus (about 250M words). The model contains 185K words.",
                "preprocessing": "The corpus was lemmatized and tagged with Universal PoS",
                "read_more": [
                    "https://www.academia.edu/24306935/WebVectors_a_Toolkit_for_Building_Web_Interfaces_for_Vector_Semantic_Models",
                    "http://rusvectores.org/en/",
                    "https://github.com/RaRe-Technologies/gensim-data/issues/3"
                ],
                "checksum": "9bdebdc8ae6d17d20839dd9b5af10bc4",
                "file_name": "word2vec-ruscorpora-300.gz",
                "parts": 1
            },
            "word2vec-google-news-300": {
                "num_records": 3000000,
                "file_size": 1743563840,
                "base_dataset": "Google News (about 100 billion words)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/__init__.py",
                "license": "not found",
                "parameters": {
                    "dimension": 300
                },
                "description": "Pre-trained vectors trained on a part of the Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases. The phrases were obtained using a simple data-driven approach described in 'Distributed Representations of Words and Phrases and their Compositionality' (https://code.google.com/archive/p/word2vec/).",
                "read_more": [
                    "https://code.google.com/archive/p/word2vec/",
                    "https://arxiv.org/abs/1301.3781",
                    "https://arxiv.org/abs/1310.4546",
                    "https://www.microsoft.com/en-us/research/publication/linguistic-regularities-in-continuous-space-word-representations/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F189726%2Frvecs.pdf"
                ],
                "checksum": "a5e5354d40acb95f9ec66d5977d140ef",
                "file_name": "word2vec-google-news-300.gz",
                "parts": 1
            },
            "glove-wiki-gigaword-50": {
                "num_records": 400000,
                "file_size": 69182535,
                "base_dataset": "Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 50
                },
                "description": "Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-50.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "c289bc5d7f2f02c6dc9f2f9b67641813",
                "file_name": "glove-wiki-gigaword-50.gz",
                "parts": 1
            },
            "glove-wiki-gigaword-100": {
                "num_records": 400000,
                "file_size": 134300434,
                "base_dataset": "Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 100
                },
                "description": "Pre-trained vectors based on Wikipedia 2014 + Gigaword 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-100.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "40ec481866001177b8cd4cb0df92924f",
                "file_name": "glove-wiki-gigaword-100.gz",
                "parts": 1
            },
            "glove-wiki-gigaword-200": {
                "num_records": 400000,
                "file_size": 264336934,
                "base_dataset": "Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-200/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 200
                },
                "description": "Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-200.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "59652db361b7a87ee73834a6c391dfc1",
                "file_name": "glove-wiki-gigaword-200.gz",
                "parts": 1
            },
            "glove-wiki-gigaword-300": {
                "num_records": 400000,
                "file_size": 394362229,
                "base_dataset": "Wikipedia 2014 + Gigaword 5 (6B tokens, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-300/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 300
                },
                "description": "Pre-trained vectors based on Wikipedia 2014 + Gigaword, 5.6B tokens, 400K vocab, uncased (https://nlp.stanford.edu/projects/glove/).",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-wiki-gigaword-300.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "29e9329ac2241937d55b852e8284e89b",
                "file_name": "glove-wiki-gigaword-300.gz",
                "parts": 1
            },
            "glove-twitter-25": {
                "num_records": 1193514,
                "file_size": 109885004,
                "base_dataset": "Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 25
                },
                "description": "Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/).",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-25.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "50db0211d7e7a2dcd362c6b774762793",
                "file_name": "glove-twitter-25.gz",
                "parts": 1
            },
            "glove-twitter-50": {
                "num_records": 1193514,
                "file_size": 209216938,
                "base_dataset": "Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-50/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 50
                },
                "description": "Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/)",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-50.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "c168f18641f8c8a00fe30984c4799b2b",
                "file_name": "glove-twitter-50.gz",
                "parts": 1
            },
            "glove-twitter-100": {
                "num_records": 1193514,
                "file_size": 405932991,
                "base_dataset": "Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-100/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 100
                },
                "description": "Pre-trained vectors based on  2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/)",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-100.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "b04f7bed38756d64cf55b58ce7e97b15",
                "file_name": "glove-twitter-100.gz",
                "parts": 1
            },
            "glove-twitter-200": {
                "num_records": 1193514,
                "file_size": 795373100,
                "base_dataset": "Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased)",
                "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-200/__init__.py",
                "license": "http://opendatacommons.org/licenses/pddl/",
                "parameters": {
                    "dimension": 200
                },
                "description": "Pre-trained vectors based on 2B tweets, 27B tokens, 1.2M vocab, uncased (https://nlp.stanford.edu/projects/glove/).",
                "preprocessing": "Converted to w2v format with `python -m gensim.scripts.glove2word2vec -i <fname> -o glove-twitter-200.txt`.",
                "read_more": [
                    "https://nlp.stanford.edu/projects/glove/",
                    "https://nlp.stanford.edu/pubs/glove.pdf"
                ],
                "checksum": "e52e8392d1860b95d5308a525817d8f9",
                "file_name": "glove-twitter-200.gz",
                "parts": 1
            },
            "__testing_word2vec-matrix-synopsis": {
                "description": "[THIS IS ONLY FOR TESTING] Word vecrors of the movie matrix.",
                "parameters": {
                    "dimensions": 50
                },
                "preprocessing": "Converted to w2v using a preprocessed corpus. Converted to w2v format with `python3.5 -m gensim.models.word2vec -train <input_filename> -iter 50 -output <output_filename>`.",
                "read_more": [],
                "checksum": "534dcb8b56a360977a269b7bfc62d124",
                "file_name": "__testing_word2vec-matrix-synopsis.gz",
                "parts": 1
            }
        }
    }




There are two types of data resources: corpora and models.


.. code-block:: default

    print(info.keys())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    dict_keys(['corpora', 'models'])




Let's have a look at the available corpora:


.. code-block:: default

    for corpus_name, corpus_data in sorted(info['corpora'].items()):
        print(
            '%s (%d records): %s' % (
                corpus_name,
                corpus_data.get('num_records', -1),
                corpus_data['description'][:40] + '...',
            )
        )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    20-newsgroups (18846 records): The notorious collection of approximatel...
    __testing_matrix-synopsis (-1 records): [THIS IS ONLY FOR TESTING] Synopsis of t...
    __testing_multipart-matrix-synopsis (-1 records): [THIS IS ONLY FOR TESTING] Synopsis of t...
    fake-news (12999 records): News dataset, contains text and metadata...
    patent-2017 (353197 records): Patent Grant Full Text. Contains the ful...
    quora-duplicate-questions (404290 records): Over 400,000 lines of potential question...
    semeval-2016-2017-task3-subtaskA-unannotated (189941 records): SemEval 2016 / 2017 Task 3 Subtask A una...
    semeval-2016-2017-task3-subtaskBC (-1 records): SemEval 2016 / 2017 Task 3 Subtask B and...
    text8 (1701 records): First 100,000,000 bytes of plain text fr...
    wiki-english-20171001 (4924894 records): Extracted Wikipedia dump from October 20...




... and the same for models:


.. code-block:: default

    for model_name, model_data in sorted(info['models'].items()):
        print(
            '%s (%d records): %s' % (
                model_name,
                model_data.get('num_records', -1),
                model_data['description'][:40] + '...',
            )
        )





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    __testing_word2vec-matrix-synopsis (-1 records): [THIS IS ONLY FOR TESTING] Word vecrors ...
    conceptnet-numberbatch-17-06-300 (1917247 records): ConceptNet Numberbatch consists of state...
    fasttext-wiki-news-subwords-300 (999999 records): 1 million word vectors trained on Wikipe...
    glove-twitter-100 (1193514 records): Pre-trained vectors based on  2B tweets,...
    glove-twitter-200 (1193514 records): Pre-trained vectors based on 2B tweets, ...
    glove-twitter-25 (1193514 records): Pre-trained vectors based on 2B tweets, ...
    glove-twitter-50 (1193514 records): Pre-trained vectors based on 2B tweets, ...
    glove-wiki-gigaword-100 (400000 records): Pre-trained vectors based on Wikipedia 2...
    glove-wiki-gigaword-200 (400000 records): Pre-trained vectors based on Wikipedia 2...
    glove-wiki-gigaword-300 (400000 records): Pre-trained vectors based on Wikipedia 2...
    glove-wiki-gigaword-50 (400000 records): Pre-trained vectors based on Wikipedia 2...
    word2vec-google-news-300 (3000000 records): Pre-trained vectors trained on a part of...
    word2vec-ruscorpora-300 (184973 records): Word2vec Continuous Skipgram vectors tra...




If you want to get detailed information about a model/corpus, use:



.. code-block:: default



    fake_news_info = api.info('fake-news')
    print(json.dumps(fake_news_info, indent=4))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {
        "num_records": 12999,
        "record_format": "dict",
        "file_size": 20102776,
        "reader_code": "https://github.com/RaRe-Technologies/gensim-data/releases/download/fake-news/__init__.py",
        "license": "https://creativecommons.org/publicdomain/zero/1.0/",
        "fields": {
            "crawled": "date the story was archived",
            "ord_in_thread": "",
            "published": "date published",
            "participants_count": "number of participants",
            "shares": "number of Facebook shares",
            "replies_count": "number of replies",
            "main_img_url": "image from story",
            "spam_score": "data from webhose.io",
            "uuid": "unique identifier",
            "language": "data from webhose.io",
            "title": "title of story",
            "country": "data from webhose.io",
            "domain_rank": "data from webhose.io",
            "author": "author of story",
            "comments": "number of Facebook comments",
            "site_url": "site URL from BS detector",
            "text": "text of story",
            "thread_title": "",
            "type": "type of website (label from BS detector)",
            "likes": "number of Facebook likes"
        },
        "description": "News dataset, contains text and metadata from 244 websites and represents 12,999 posts in total from a specific window of 30 days. The data was pulled using the webhose.io API, and because it's coming from their crawler, not all websites identified by their BS Detector are present in this dataset. Data sources that were missing a label were simply assigned a label of 'bs'. There are (ostensibly) no genuine, reliable, or trustworthy news sources represented in this dataset (so far), so don't trust anything you read.",
        "checksum": "5e64e942df13219465927f92dcefd5fe",
        "file_name": "fake-news.gz",
        "read_more": [
            "https://www.kaggle.com/mrisdal/fake-news"
        ],
        "parts": 1
    }




Sometimes, you do not want to load a model into memory. Instead, you can request
just the filesystem path to the model. For that, use:



.. code-block:: default



    print(api.load('glove-wiki-gigaword-50', return_path=True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/kofola3/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz




If you want to load the model to memory, then:



.. code-block:: default



    model = api.load("glove-wiki-gigaword-50")
    model.most_similar("glass")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [('plastic', 0.79425048828125), ('metal', 0.7708716988563538), ('walls', 0.7700635194778442), ('marble', 0.7638523578643799), ('wood', 0.7624280452728271), ('ceramic', 0.7602593302726746), ('pieces', 0.7589112520217896), ('stained', 0.7528817653656006), ('tile', 0.748193621635437), ('furniture', 0.7463858723640442)]



For corpora, the corpus is never loaded to memory, all corpora are iterables wrapped in
a special class ``Dataset``, with an ``__iter__`` method.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  39.422 seconds)

**Estimated memory usage:**  297 MB


.. _sphx_glr_download_auto_examples_howtos_run_downloader_api.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: run_downloader_api.py <run_downloader_api.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: run_downloader_api.ipynb <run_downloader_api.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
