#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
CLI USAGE: python glove2word2vec.py -input <GloVe vector file> -output <Output model file>

Convert GloVe vectors into word2vec C format;

model = gensim.models.Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format

word2vec embeddings start with a line with the number of lines (tokens?) and the number of dimensions of the file. This allows gensim to allocate memory 
accordingly for querying the model. Larger dimensions mean larger memory is held captive. Accordingly, this line has to be inserted into the GloVe 
embeddings file.
"""


import re
import sys
import gensim
import logging
import argparse
import smart_open
import numpy as np

logger = logging.getLogger(__name__)

def glove2word2vec(glove_vector_file, output_model_file):
    """Convert GloVe vectors into word2vec C format"""
    
    def count_dims(filename):
        """ 
        Function to calculate the number of dimensions from an embeddings file
        """
        count=0
        dims=[]
        for line in smart_open.smart_open(filename):
            count+=1
            if count<100:
                dims.append(len(re.findall('[\d]+.[\d]+', line)))
            else: break
            return int(np.median(dims))

    def get_info(glove_file_name):
        """ 
        Function to calculate the number of lines and dimensions of the GloVe vectors to make it Gensim compatible
        """
        num_lines = sum(1 for line in smart_open.smart_open(glove_vector_file))
        dims= count_dims(glove_file_name)
        return num_lines, dims
    
    def prepend_line(infile, outfile, line):
        """ 
        Function to prepend lines using smart_open
        """
        with smart_open.smart_open(infile, 'rb') as old:
            with smart_open.smart_open(outfile, 'wb') as new:
                new.write(str(line) + " \n ")
                for line in old:
                    new.write(line)
        return outfile
        
    num_lines, dims= get_info(glove_vector_file)
    gensim_first_line = "{} {}".format(num_lines, dims)
    model_file=prepend_line(glove_vector_file, output_model_file, gensim_first_line)

    model=gensim.models.Word2Vec.load_word2vec_format(model_file, binary=False)
    return num_lines, dims, model, model_file

if __name__ == "__main__":
    
    program = sys.argv[0]
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-input", help="Use GloVe model file to convert into word2vec C format", type=str, required=True)
    parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors", type=str, default='output_vectors.txt')

    
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ' '.join(sys.argv))
    
    args = parser.parse_args()

    glove_vector_file= args.input
    output_model_file= args.output
    
    
    num_lines, dims, model, model_file= glove2word2vec(glove_vector_file, output_model_file)

    logger.info('%d lines with %d dimensions', num_lines, dims)
    logger.info('Model %s successfully created', output_model_file)
    
    logger.info('Testing the model....')
    logger.info('Most similar to king are:%s ', model.most_similar(positive=['king'], topn=10))
    logger.info('Similarity score between woman and man is %s ', model.similarity('woman', 'man'))
    
    logger.info("Finished running %s", program)
