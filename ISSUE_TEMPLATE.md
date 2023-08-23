<!--
**IMPORTANT**:

- Use the [Gensim mailing list](https://groups.google.com/g/gensim) to ask general or usage questions. Github issues are only for bug reports.
- Check [Recipes&FAQ](https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ) first for common answers.

Github bug reports that do not include relevant information and context will be closed without an answer. Thanks!
-->

#### Problem description

What are you trying to achieve? What is the expected result? What are you seeing instead?

#### Steps/code/corpus to reproduce

Include full tracebacks, logs and datasets if necessary. Please keep the examples minimal ("minimal reproducible example").

If your problem is with a specific Gensim model (word2vec, lsimodel, doc2vec, fasttext, ldamodel etc), include the following:

```python
print(my_model.lifecycle_events)
```

#### Versions

Please provide the output of:

```python
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import struct; print("Bits", 8 * struct.calcsize("P"))
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import gensim; print("gensim", gensim.__version__)
from gensim.models import word2vec;print("FAST_VERSION", word2vec.FAST_VERSION)
```
