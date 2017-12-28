#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Create data
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# Draw graoh
plt.plot(x, y)
plt.show()
