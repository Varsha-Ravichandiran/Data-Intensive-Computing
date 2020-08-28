#!/usr/bin/env python3

import sys
import numpy
from numpy import genfromtxt
from math import sqrt

test_data = genfromtxt('Test_norm.csv', delimiter=',')
test_data = numpy.asarray(test_data, dtype=numpy.float32)
for lines in sys.stdin:
    lines = lines.strip()
    data = lines.split(',')
    train_data = data[:48]
    y = lines.split(',')[-1]
    train_row = numpy.asarray(train_data, dtype=numpy.float32)
    for test_row in test_data:
        euclidean_dist = 0
        for i in range(len(test_row) - 1):
            euclidean_dist += (test_row[i] - train_row[i]) ** 2
        dist = sqrt(euclidean_dist)
        res = ','.join(map(str, test_row))
        print('{0}\t{1}\t{2}'. format(res, str(y), str(dist)))
