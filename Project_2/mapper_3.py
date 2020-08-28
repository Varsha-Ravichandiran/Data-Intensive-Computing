import sys
import os
import re
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
nltk.download('stopwords')

ps = PorterStemmer() 
stop_words = set(stopwords.words('english'))
for lines in sys.stdin:
	lines = lines.strip()
	words = lines.split()
	path = os.getenv('map_input_file')
	head, docid = os.path.split(path)
	for text in words:
		word = re.sub('[^a-zA-Z\s]','',text)
		word = word.lower()
		word = ps.stem(word)
		if word not in stop_words:
			print('{0}\t{1}'. format(word, docid))
