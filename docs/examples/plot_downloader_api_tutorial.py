# -*- coding: utf-8 -*-
"""
EXAMPLE EXAMPLE
===============

TO BE REPLACED WITH A REAL EXAMPLE
"""

###########################################################################
# Here's our imports:

import numpy as np
import matplotlib.pyplot as plt

###########################################################################
# Here's the first graph:

x = np.linspace(-1, 2, 100)
y = np.exp(x)

plt.figure()
plt.plot(x, y)
plt.xlabel('$x$')
plt.ylabel('$\exp(x)$')

plt.show()

###########################################################################
# And here's the second one:

plt.figure()
plt.plot(x, -np.exp(-x))
plt.xlabel('$x$')
plt.ylabel('$-\exp(-x)$')

plt.show()
