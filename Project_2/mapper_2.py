#!/usr/bin/env python3

import sys
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
keys = ["science", "sea", "fire"]

for text in sys.stdin:
    text = text.lower()
    text = re.sub('[\n]', ' ', text)
    text = re.sub('[^a-z0-9 .?!]', '', text)
    lines = re.split('[.?!]', text)
    for line in lines:
        words = []
        for word in line.strip().split():
            if word not in stop_words:
                words.append(word)

        if len(words) >= 3:
            for i in range(len(words)-2):
                for k in keys:
                    if k in words[i]:
                        print(line + '\t' + '$_' + words[i+1] + '_' + words[i+2] + '\t' + '1')
                    elif k in words[i+1]:
                        print(line + '\t' + words[i] + '_$_' + words[i+2] + '\t' + '1')
                    elif k in words[i+2]:
                        print(line + '\t' + words[i] + '_' + words[i+1] + '_$' + '\t' + '1')

