import numpy as np


path = './../wsdream/dataset2/'

rt = []
with open(path+'rtdata.txt', 'r') as f:
    for line in f:
        temp = list(map(float, line.strip().split(' ')))
        rt.append(temp)

rt = np.array(rt)
print(rt.shape)

uid = np.unique(rt[:, :1])
sid = np.unique(rt[:, 1:2])
tid = np.unique(rt[:, 2:3])

print(uid.shape)
print(sid.shape)
print(tid.shape)
