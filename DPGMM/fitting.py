#!/usr/bin/python
# -*- coding: utf-8 -*-
# original: http://nbviewer.ipython.org/github/breakbee/PyNote/blob/master/Implementation_of_DPGMM.ipynb
# require python 3.2

import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from dpgmm import *
import functools
import pandas

# データ次元
dim = 2

# データのロード
data_frame = pandas.read_csv('./old_faithful.dat')
X = data_frame.as_matrix()

# DP-GMMによるパラメタ推定
iteration = 50
dpgmm = DPGMM()
X_s = dpgmm.fit(X, iteration)
