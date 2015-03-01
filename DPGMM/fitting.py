#!/usr/bin/python
# -*- coding: utf-8 -*-
# original: http://nbviewer.ipython.org/github/breakbee/PyNote/blob/master/Implementation_of_DPGMM.ipynb
# require python 3.2

from dpgmm import *
import pandas

# データのロード
data_frame = pandas.read_csv('./old_faithful.dat')
X = data_frame.as_matrix()

# DP-GMMによるパラメタ推定
iteration = 50
dpgmm = DPGMM()
X_s = dpgmm.fit(X, iteration)
