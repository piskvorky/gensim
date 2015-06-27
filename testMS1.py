# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:45:29 2015

@author: ken
"""

from gensim.models import Word2Vec
import time
import pandas as pd

googVecFilePath = "/home/ken/Word2VecPreTrainedVectors/GoogleNews-vectors-negative300.bin.gz"

def loadGoogModel():
    googModel = Word2Vec.load_word2vec_format(googVecFilePath, binary=True)
    return googModel


import sys
if sys.version[0] == "2":
    import StringIO as io
else:
    import io

import cProfile, pstats


def testMS1_void(model):
    start = time.time()

    for i in range(10):
        results_list = model.most_similar("catfish")

    end = time.time()

    pr = cProfile.Profile()
    pr.enable()

    for i in range(10):
        results_list = model.most_similar("catfish")

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats()


    with open("/home/ken/Dropbox/McKinsey/Python/Github/gensim/KenTests/orig.txt", 'w') as f:
        for line in results_list:
            f.write(line[0] + "\t" + str(line[1]) + "\n")

        f.write("\n10 runs in " + str(end-start) + " seconds\n\n\n\n")

        f.write(s.getvalue())


def getSectionReport_df(sections):
    sectionReport_list = []
    for section in sections:
        sectionReport_list.append((section["section"], len(section["correct"]), len(section["incorrect"])))

    sectionReport_df = pd.DataFrame(sectionReport_list, columns=['Section', 'Correct', 'Incorrect'])
    sectionReport_df["PercentCorrect"] = sectionReport_df['Correct'] / (sectionReport_df['Correct'] + sectionReport_df['Incorrect'])
    return sectionReport_df


def timeAccuracy(model, version):
    start = time.time()

    if '1' in version:
        sections = model.accuracy('/home/ken/Dropbox/McKinsey/Python/Github/gensim/questions-words.txt')
    elif '2' in version:
        sections = model.accuracy_v2('/home/ken/Dropbox/McKinsey/Python/Github/gensim/questions-words.txt')

    stop = time.time()

    sectionReport_df = getSectionReport_df(sections)

    sectionReport_df["FullTime"] = stop-start

    sectionReport_df.to_csv('/home/ken/Dropbox/McKinsey/Python/Github/gensim/KenTests/sectionReport_' + version + '_df.csv')

if __name__ == '__main__':
    print("Starting v2...")
    googModel = loadGoogModel()
    t1 = time.time()
    timeAccuracy(googModel, 'v2')
    t2 = time.time()
    print("Finished v2 in roughly " + str(t2-t1) + " seconds")
    timeAccuracy(googModel, 'v1')
    t3 = time.time()
    print("Finished v1 in roughly " + str(t3-t2) + " seconds")

