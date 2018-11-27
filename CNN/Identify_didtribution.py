import numpy as np
rt = []
with open("./../Input/rtdata.txt",'r') as fl:
	for line in fl:
		rt.append(float(line.strip().split(' ')[3]))

rt.sort(reverse = True)
print rt[int(len(rt)*0.3)]
print rt[int(len(rt)*0.5)]
print rt[int(len(rt)*0.7)]
print rt[int(len(rt)*0.9)]



