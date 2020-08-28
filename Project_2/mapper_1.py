#!/usr/bin/env python3
import sys
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

ps=PorterStemmer()
stop_words = set(stopwords.words('english'))
for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        text = re.sub('[^a-zA-Z\s]','',word)
        text = text.lower()
        text = ps.stem(text)
        if text:
            if text not in stop_words:            
                print ('%s\t%s' % (text, "1"))

