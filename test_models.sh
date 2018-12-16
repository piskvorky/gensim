#!/usr/bin/env bash

# Run this file to check for already built models
# The vectors for the words should match

# Example output:
#----------------
#Checking Only Unicode file for малинка
#FB FastText vector for 'малинка':
#малинка 0.0016197 0.0006822 -0.00079548 3.4498e-05 -0.00041843 0.0016217 -0.0019639 0.0018944 0.00015089 -0.0018636 0.0017409 0.00041354 5.2017e-05 0.00057594 -0.0014678 0.00064968 -0.00043663 -0.0013571 -0.00075255 -0.00064337 -0.00014929 -8.5475e-06 -0.00090269 -0.00062622 0.0010743 -0.0002109 0.00051223 -0.00016063 0.00099641 0.00085189 0.00062887 -0.001691 -0.0021112 -0.00068603 0.0017625 0.00034326 -0.0017784 8.9783e-05 0.0002942 0.00096906 -0.00080397 -0.0010684 0.0011714 -0.0014452 0.00079747 0.0014469 -0.00068339 0.00026686 -7.5541e-05 0.00010066 -0.0023944 -0.001295 -0.00057308 -0.00040282 0.00021395 -0.0038837 0.0023913 0.00046846 0.0025767 2.6313e-05 -0.001706 0.00022219 -1.6816e-05 0.0022659 -0.00063133 6.8836e-05 0.0021171 -8.8353e-05 -0.0015784 0.0012827 0.0018913 0.0028348 0.0010575 0.0016142 0.002166 0.00069436 -0.0013646 -0.0014233 -0.0011904 -7.5761e-05 -0.0016289 2.7371e-05 -0.0015946 0.00093957 0.00133 0.00043247 0.000146 -0.00027774 -0.0034843 0.00010504 -0.00030653 0.00032292 0.00084782 5.9449e-05 -0.0010364 -0.0024459 0.0023318 -0.00042711 -0.00042472 -0.00044134
#Gensim recreated vector for 'малинка':
#малинка <class 'str'>
#INFO:gensim.models.fasttext:loading 41 words for fastText model from only_unicode_ft_model.bin
#INFO:gensim.models.fasttext:loading weights for 41 words for fastText model from only_unicode_ft_model.bin
#INFO:gensim.models.fasttext:loaded (41, 100) weight matrix for fastText model from only_unicode_ft_model.bin
#[ 1.6196765e-03  6.8219670e-04 -7.9547794e-04  3.4498495e-05
# -4.1842801e-04  1.6216735e-03 -1.9638820e-03  1.8944255e-03
#  1.5088824e-04 -1.8635745e-03  1.7408876e-03  4.1354020e-04
#  5.2017498e-05  5.7593960e-04 -1.4678127e-03  6.4967724e-04
# -4.3663362e-04 -1.3571251e-03 -7.5255416e-04 -6.4337463e-04
# -1.4928996e-04 -8.5474967e-06 -9.0269221e-04 -6.2622159e-04
#  1.0742976e-03 -2.1089800e-04  5.1222846e-04 -1.6062708e-04
#  9.9641387e-04  8.5188699e-04  6.2887446e-04 -1.6909525e-03
# -2.1112466e-03 -6.8602891e-04  1.7625110e-03  3.4326490e-04
# -1.7784367e-03  8.9782574e-05  2.9419785e-04  9.6906058e-04
# -8.0397044e-04 -1.0684382e-03  1.1713610e-03 -1.4451878e-03
#  7.9746643e-04  1.4468502e-03 -6.8339315e-04  2.6686297e-04
# -7.5540847e-05  1.0065706e-04 -2.3943805e-03 -1.2949962e-03
# -5.7307532e-04 -4.0282018e-04  2.1394751e-04 -3.8837022e-03
#  2.3913269e-03  4.6845980e-04  2.5766643e-03  2.6313142e-05
# -1.7059513e-03  2.2218851e-04 -1.6815799e-05  2.2658831e-03
# -6.3132925e-04  6.8836329e-05  2.1171137e-03 -8.8352586e-05
# -1.5783844e-03  1.2827339e-03  1.8913266e-03  2.8348232e-03
#  1.0574928e-03  1.6142147e-03  2.1659583e-03  6.9436466e-04
# -1.3646375e-03 -1.4233063e-03 -1.1903865e-03 -7.5761469e-05
# -1.6289009e-03  2.7370954e-05 -1.5945622e-03  9.3956792e-04
#  1.3300242e-03  4.3246627e-04  1.4599577e-04 -2.7774350e-04
# -3.4842787e-03  1.0503911e-04 -3.0653202e-04  3.2292373e-04
#  8.4782037e-04  5.9448710e-05 -1.0364439e-03 -2.4459062e-03
#  2.3318187e-03 -4.2710797e-04 -4.2471924e-04 -4.4134055e-04]
#------------------------------


echo "------------------------------"
echo "Checking Non Unicode file for word 'chained' "
echo "FB FastText vector for 'chained': "
cat no_unicode_ft_model.vec | grep chained
echo "Gensim recreated vector for 'chained': "
python check_ft_model.py no_unicode_ft_model.bin chained
echo "------------------------------"
echo ""

echo "------------------------------"
echo "Checking Only Unicode file for малинка"
echo "FB FastText vector for 'малинка': "
cat only_unicode_ft_model.vec | grep "малинка "
echo "Gensim recreated vector for 'малинка': "
python3 check_ft_model.py only_unicode_ft_model.bin малинка
echo "------------------------------"
echo ""

echo "------------------------------"
echo "Checking Mixed file for малинка"
echo "FB FastText vector for 'малинка': "
cat mix_unicode_ft_model.vec | grep "малинка "
echo "Gensim recreated vector for 'малинка': "
python3 check_ft_model.py only_unicode_ft_model.bin малинка
echo "------------------------------"
echo ""

echo "------------------------------"
echo "Checking Mixed file for chained"
echo "FB FastText vector for 'chained': "
cat mix_unicode_ft_model.vec | grep "chained"
echo "Gensim recreated vector for 'chained': "
python3 check_ft_model.py mix_unicode_ft_model.bin chained
echo "------------------------------"
echo ""