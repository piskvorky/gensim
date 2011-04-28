import os
modules_path = os.path.join('site-packages', 'gensim', 'src', 'gensim')
__path__.append(os.path.join(os.path.dirname(os.__file__), modules_path))
