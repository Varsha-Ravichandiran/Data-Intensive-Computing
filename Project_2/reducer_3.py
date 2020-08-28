import sys

inverted_index = {}
for line in sys.stdin:
	line = line.rstrip('\n\n')
	word, docid = line.split('\t')
	if word not in inverted_index:
		inverted_index[word] = [docid]
	else:
		if docid not in inverted_index[word]:
			inverted_index[word].append(docid)
for word in inverted_index:
	print(word, inverted_index[word])
