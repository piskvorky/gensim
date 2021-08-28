from poincare import PoincareModel, PoincareRelations
from time import time
import numpy as np
t1 = time()
file_path = "C:\\Users\\sagar\\gensim\\gensim\\test\\test_data\\poincare_hypernyms_large.tsv"
model = PoincareModel(PoincareRelations(file_path), negative=2)
model.train(epochs=50)
t2 = time()
print(t2-t1)
#print((np.random.randint.__doc__))


print(np.random.RandomState.rand.__doc__)
print(np.random.default_rng(1).gamma.__doc__)