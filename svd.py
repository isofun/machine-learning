#-*- coding: utf-8 -*-
import numpy as np

dataList = []
labelListTr = []
trainName = input('input the training file name: ')
f = open(trainName, 'r')
for line in f.readlines():
	temp = line.strip().split(',')
	data = [float(x) for x in temp[0: len(temp) - 1]]
	label = int(temp[-1])
	dataList.append(data)
	labelListTr.append(label)
f.close
print('size of training matrix is %d * %d' % (len(dataList), len(dataList[0])))

testList = []
labelListTe = []
testName = input('input the testing file name: ')
f = open(testName, 'r')
for line in f.readlines():
	temp = line.strip().split(',')
	data = [float(x) for x in temp[0: len(temp) - 1]]
	label = int(temp[-1])
	testList.append(data)
	labelListTe.append(label)
f.close()
print('size of testing matrix is %d * %d' % (len(testList), len(testList[0])))

u1, s1, vt1 = np.linalg.svd(dataList)
u2, s2, vt2 = np.linalg.svd(testList)
for k in [10, 20, 30]:
#	u1 = np.dot(np.mat(u1[:, :k]),np.diag(s1[:k]))
	u1 = np.dot(np.array(dataList), vt1[:,:k])
#	u2 = np.dot(np.mat(u2[:, :k]),np.diag(s2[:k]))
	u2 = np.dot(np.array(testList), vt2[:,:k])
	count = 0
	for i in range(u2.shape[0]):
		dis = [np.sqrt(sum((u2[i] - line)**2)) for line in u1]
		ind = dis.index(min(dis))
		if labelListTe[i] == labelListTr[ind]:
			count = count + 1
	pred = float(count / u2.shape[0])
	print(pred)


