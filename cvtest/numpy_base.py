#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:09:25 2020

@author: bright
"""

import numpy as np
from numpy import pi

a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
print(a.ndim)
print(a.itemsize)
a = np.array([1,2,3,4], dtype=np.float64) 
b = np.array([(1.5,2,3), (4,5,6)], dtype=np.int32)
c = np.array( [ [1,2], [3,4] ], dtype=np.complex128 )

d=np.linspace( 0, 2, 9 )
s = "d = " + repr(d)
print(s)
x = np.linspace( 0, 2*pi, 100 )
f = np.sin(x)
s = "f = " + repr(f)
print(s)

#products
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
c = a*b
d = a.dot(b)
s = "c = " + repr(c) + "\nd = " + repr(d)
print(s)
