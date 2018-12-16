#!/usr/bin/env bash

# Gets all the needed code and build the fast text binary using files in
# gensim/test/test_data
#
# Run this file once. Once this is run, you should have the models built in the
# current directory

echo "Cloning fastText from FB github repo"
git clone https://github.com/facebookresearch/fastText.git
echo "Clone complete"

FB_FT_PATH="`pwd`/fastText"
echo "fastText path is " $FB_FT_PATH

echo "Making fastText binary. Make sure 'make' is installed"
cd $FB_FT_PATH
make
echo "fastText build complete"
cd ../

echo "Making no unicode model"
fastText/fasttext skipgram -minCount 0 -input gensim/test/test_data/ft_test_data/no_unicode.txt -output no_unicode_ft_model

echo "Making only unicode model"
fastText/fasttext skipgram -minCount 0 -input gensim/test/test_data/ft_test_data/only_unicode.txt -output only_unicode_ft_model

echo "Making only unicode non-unicode model"
fastText/fasttext skipgram -minCount 0 -input gensim/test/test_data/ft_test_data/unicode_non_unicode_mix.txt -output mix_unicode_ft_model