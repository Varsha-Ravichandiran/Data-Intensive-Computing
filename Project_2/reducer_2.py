#!/usr/bin/env python3

import sys
import heapq

f = {}
for s in sys.stdin:
    try:
        _, tgram, count = s.strip().split('\t')
        count = int(count)
        f[tgram] += count
    except KeyError:
        f[tgram] = count
    except ValueError:
        continue
q = []
for k,v in f.items():
    if len(q) < 10:
        heapq.heappush(q, (v, k))
    else:
        if v > q[0][0]:
            heapq.heappop(q)
            heapq.heappush(q, (v,k))
while q:
    pop = heapq.heappop(q)
    print(pop[1] + '\t' + str(pop[0]))
