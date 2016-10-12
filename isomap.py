#-*- coding: utf-8 -*-
import numpy as np

def readdata(name):
	dataList = []
	labelList = []
#	name = input('input the training file name: ')
#	name = 'sonar-train.txt'
	f = open(name, 'r')
	for line in f.readlines():
		temp = line.strip().split(',')
		data = [float(x) for x in temp[0: len(temp) - 1]]
		label = int(temp[-1])
		dataList.append(data)
		labelList.append(label)
	f.close
	print('size of training matrix is %d * %d' % (len(dataList), len(dataList[0])))
	return dataList, labelList

#testList = []
#labelListTe = []
#testName = input('input the testing file name: ')
#f = open(testName, 'r')
#for line in f.readlines():
#	temp = line.strip().split(',')
#	data = [float(x) for x in temp[0: len(temp) - 1]]
#	label = int(temp[-1])
#	testList.append(data)
#	labelListTe.append(label)
#f.close()
#print('size of testing matrix is %d * %d' % (len(testList), len(testList[0])))

def mds(matrix, k):
	D1 = []
	for line1 in matrix:
		D1.append([np.sqrt(sum((np.array(line1) - np.array(line2))**2)) for line2 in matrix])
	D12 = np.array(D1)**2
	N = len(D1)
	H = np.eye(N) - 1/N
	T = -0.5 * np.dot(np.dot(H, D12), H)
	eigVal, eigVect = np.linalg.eig(T)
	newMat = np.dot(np.array(eigVect[:, :k]), np.diag(np.sqrt(eigVal[:k])))
	return newMat

def knn(matrix, k):
	ret = np.ones((len(matrix), len(matrix))) * 65535
	for i in range(len(matrix)):
		dis = [float(np.sqrt((np.mat(matrix[i]) - np.mat(line)) * ((np.mat(matrix[i]) - np.mat(line)).T))) for line in matrix]
		ind = np.argsort(dis)[1:k+1]
		for j in ind:
			ret[i][j] = dis[j]
	return [dijkstra(ret,i) for i in range(len(ret))]

def dijkstra(matrix, v):
	D = [matrix[v][i] for i in range(len(matrix))]
	D[v] = 0
	set1 = set({})
	set2 = set([i for i in range(len(matrix))])
	set1.add(v)
	set2.remove(v)
	while(len(set1) != len(matrix)):
		mini = min(D[i] for i in set2)
		for i in set2:
			if D[i] == mini:
				flag = i
				break
		set1.add(flag)
		set2.remove(flag)
		D[flag] = mini
		for i in set2:
			D[i] = min(D[i], D[flag] + matrix[flag][i])
	return D

if __name__ == '__main__':
	trainList, trainLabel = readdata(input('input the training file name: '))
	testList, testLabel = readdata(input('input the testing file name: '))
	newTrain = knn(trainList, 4)
	newTest = knn(testList, 4)
	for k in [10, 20, 30]:
		trainMds = mds(newTrain, k)
		testMds = mds(newTest, k)
		count = 0
		for i in range(len(testMds)):
			dis = [np.sqrt(sum((np.array(testMds[i]) - np.array(line))**2)) for line in trainMds]
			ind = dis.index(min(dis))
			if testLabel[i] == trainLabel[ind]:
				count = count + 1
		pred = float(count / len(newTrain))
		print(pred)
