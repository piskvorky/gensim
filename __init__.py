# script by dedan: helps him to symlink gensim
import os
modules_path = os.path.join('site-packages', 'gensim', 'gensim')
__path__.append(os.path.join(os.path.dirname(os.__file__), modules_path))
