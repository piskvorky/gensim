#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the MIT License https://opensource.org/licenses/MIT

"""
CLI USAGE: python glove2word2vec.py <GloVe vector file> <Output model file>

Convert GloVe vectors into Gensim compatible format to instantiate from an existing file on disk in the word2vec C format;

model = gensim.models.Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format

word2vec embeddings start with a line with the number of lines (tokens?) and the number of dimensions of the file. This allows gensim to allocate memory 
accordingly for querying the model. Larger dimensions mean larger memory is held captive. Accordingly, this line has to be inserted into the GloVe 
embeddings file.

"""
import re
import sys
import gensim
import smart_open


def glove2word2vec(glove_vector_file,output_model_file):

    
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
        return num_lines,dims
    
    def prepend_line(infile, outfile, line):
        """ 
        Function to prepend lines using smart_open
        """
        with smart_open.smart_open(infile, 'rb') as old:
            with smart_open.smart_open(outfile, 'wb') as new:
                new.write(str(line) + "\n")
                for line in old:
                    new.write(line)
        return outfile
        
    
    num_lines,dims=get_info(glove_vector_file)
    gensim_first_line = "{} {}".format(num_lines, dims)
    
    print '%s lines with %s dimensions' %(num_lines,dims)
    
    model_file=prepend_line(glove_vector_file,output_model_file,gensim_first_line)
    
    # Demo: Loads the newly created glove_model.txt into gensim API.  
    model=gensim.models.Word2Vec.load_word2vec_format(model_file,binary=False) #GloVe Model  
    print 'Most similar to king are: ', model.most_similar(positive=['king'], topn=10)
    print 'Similarity score between woman and man is: ', model.similarity('woman', 'man')
    print 'Model %s successfully created !!'%output_model_file
    
    return model_file

if __name__ == "__main__":

    glove_vector_file=sys.argv[1]
    output_model_file=sys.argv[2]
    glove2word2vec(glove_vector_file,output_model_file)
