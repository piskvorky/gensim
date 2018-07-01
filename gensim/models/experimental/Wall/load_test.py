from keras.engine.topology import Layer
import keras.backend as K
from keras.models import load_model
import sys
import os

sys.path.append(os.path.join('..'))

from custom_layers import TopKLayer
from drmm_tks import DRMM_TKS
# model = load_model('mera_model', custom_objects={'TopKLayer': TopKLayer})

q = [[0,1,2,3,4,4958,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051],
[0,1,2,3,4,4958,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051]]

d = [[400052,292,293,2,294,104,400052,400052,400052,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051],
[400052,2,294,6,16,294,4,130,11,297,
9,16,2,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051,
400051,400051,400051,400051,400051,400051,400051,400051,400051,400051]]

import numpy as np
q = np.array(q)
d = np.array(d)


# print(model.predict(
#             x={'query': q, 'doc': d}))

new_model = DRMM_TKS.load('mera_model')
print("New model ", new_model.model.predict(
            x={'query': q, 'doc': d}))
