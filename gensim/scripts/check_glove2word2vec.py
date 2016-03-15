from gensim.utils import check_output

"""
Test the command line arguments for the converter file glove2word2vec.py with a sample vector file sample_glove.txt
"""
   
print check_output(['python', 'glove2word2vec.py', '-i', 'sample_glove.txt', '-o', 'sample_word2vec_out.txt'])
print check_output(['python', 'glove2word2vec.py', '--input', 'sample_glove.txt', '--output', 'sample_word2vec_out.txt'])
