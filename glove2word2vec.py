#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
CLI USAGE: python glove2word2vec.py <GloVe vector file> <Output model file>

Convert GloVe vectors into word2vec C format for usage;
model = gensim.models.Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format

Word2vec embeddings start with a line with the number of lines (tokens?) and the number of dimensions of the file. This allows gensim to allocate memory 
accordingly for querying the model. Larger dimensions mean larger memory is held captive. Accordingly, this line has to be inserted into the GloVe 
embeddings file.
"""


import re
import sys
import gensim
import logging
import smart_open

program=sys.argv[0]
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

def glove2word2vec(glove_vector_file, output_model_file):
    """Convert GloVe vectors into word2vec C format"""
    
    def get_info(glove_file_name):
        """ 
        Function to calculate the number of lines and dimensions of the GloVe vectors to make it Gensim compatible
        """
        num_lines = sum(1 for line in smart_open.smart_open(glove_vector_file))
        if 'twitter' in glove_file_name:
            dims= re.findall('\d+',glove_vector_file.split('.')[3])
            dims=''.join(dims)
        else:
            dims=re.findall('\d+',glove_vector_file.split('.')[2])
            dims=''.join(dims)
        return num_lines, dims
    
    def prepend_line(infile, outfile, line):
        """ 
        Function to prepend lines using smart_open
        """
        with smart_open.smart_open(infile, ' rb ') as old:
            with smart_open.smart_open(outfile, ' wb ') as new:
                new.write(str(line) + " \n ")
                for line in old:
                    new.write(line)
        return outfile
        
    num_lines, dims= get_info(glove_vector_file)
    
    logger.info('%d lines with %s dimensions' %(num_lines, dims))
    
    gensim_first_line = "{} {}".format(num_lines, dims)
    model_file=prepend_line(glove_vector_file, output_model_file, gensim_first_line)
    
    logger.info('Model %s successfully created !!'%output_model_file)
    
    # Demo: Loads the newly created glove_model.txt into gensim API.  
    model=gensim.models.Word2Vec.load_word2vec_format(model_file, binary=False) #GloVe Model  

    logger.info('Most similar to king are:%s '%model.most_similar(positive=['king'], topn=10))
    logger.info('Similarity score between woman and man is %s ' %model.similarity('woman', 'man'))
    
    logger.info("Finished running %s", program)
    
    return model_file

if __name__ == "__main__":
    
    glove_vector_file=sys.argv[1]
    output_model_file=sys.argv[2]

    glove2word2vec(glove_vector_file, output_model_file)
