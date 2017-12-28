#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('test.png')
plt.imshow(img)

plt.show()
