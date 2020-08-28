#!/usr/bin/env python3

import sys
from operator import itemgetter
from collections import OrderedDict

dictionary = {}

for line in sys.stdin:
    line = line.strip()
    test_row, y, dist = line.split('\t')
    list1 = [y, dist]
    if str(test_row) not in dictionary:
        dictionary[str(test_row)] = [list1]
    else:
        dictionary[str(test_row)].append(list1)

ordered_dict = {k: sorted(v, key=lambda e: e[1]) for k, v in dictionary.items()}

for k, v in ordered_dict.items():
    neighbors = 6
    new_list = v[:6]
    labels = list(map(itemgetter(0), new_list))
    label = max(labels, key=labels.count)
    print(k, label)
