"""
Simple script to enable building a FT model from a .bin and testing it for a
word

Usage: python check_ft_model.py only_unicode_ft_model.bin малинка
Builds the model from only_unicode_ft_model.bin and queries the model for малинка
"""

import sys
from gensim.models import FastText
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    model_bin_path = sys.argv[1]
    word_to_predict = sys.argv[2]
    model = FastText.load_fasttext_format(model_bin_path)
    print(model[word_to_predict])