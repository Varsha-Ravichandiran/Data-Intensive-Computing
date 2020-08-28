#!/usr/bin/env python3
import sys
wordcount = {}
for line in sys.stdin:
    line = line.strip()
    text, count = line.split('\t', 1)
    count = int(count)
    try:
        wordcount[text] = wordcount[text]+count
    except:
        wordcount[text] = count

for text in wordcount.keys():
    print ('%s\t%s'% ( text, wordcount[text] ))
