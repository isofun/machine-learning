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

dataArray = np.array(dataList)
testArray = np.array(testList)
meanVals = np.mean(dataArray, axis = 0)
testMean = np.mean(testArray, axis = 0)
dataMat = dataArray - meanVals
testMat = testArray - testMean
covMat = np.cov(dataMat, rowvar = 0)
testCov = np.cov(testMat, rowvar = 0)
eigVals, eigVects = np.linalg.eig(np.mat(covMat)) #Compute the eigenvalues and right eigenvectors of a square array
testVals, testVects = np.linalg.eig(np.mat(testCov))
#eigValInd = np.argsort(eigVals)
#testInd = np.argsort(testVals)
for k in [10, 20, 30]:
#	newInd = eigValInd[: -(k + 1): -1]
#	newTestInd = testInd[: -(k + 1): -1]
#	newEigVects = eigVects[:, newInd]
#	newTestVects = testVects[:, newTestInd]
	newEigVects = eigVects[:,:k]
	newTestVects = testVects[:,:k]
	newMat = np.dot(dataArray, newEigVects)
	newTestMat = np.dot(testArray, newTestVects)
	count = 0
	for i in range(newTestMat.shape[0]):
		dis = [np.sqrt((newTestMat[i] - line)*((newTestMat[i]-line).T)) for line in newMat]
		ind = dis.index(min(dis))
		if labelListTe[i] == labelListTr[ind]:
			count = count + 1
	pred = float(count / newTestMat.shape[0])
	print(pred)
