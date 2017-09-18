# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)  # centroids 存放k个质心
    clusterChanged = True
    count = 0
    while clusterChanged:
        count += 1
        clusterChanged = False
        for i in range(m):
            minDist = inf;
            minIndex = -1  # minDist存放已有质心到某一点的最近距离，minIndex记录是哪个质心
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 找出每次划分聚类后的点
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 根据刚刚新划分完聚类的点计算每个簇相应的新质心
    return count, centroids, clusterAssment

def biKmeans(dataSet,k,distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit =

def draw(data, center):
    length = len(center)
    fig = plt.figure
    # 绘制原始数据的散点图
    plt.scatter(data[:, 0], data[:, 1], s=25, alpha=0.4)
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 0], center[i, 1]), xytext=(center[i, 0] + 1, center[i, 1] + 1),
                     arrowprops=dict(facecolor='red'))
    plt.show()


datMat = mat(loadDataSet('testSet2.txt'))
count, myCentroids, clustAssing = kMeans(datMat, 3)
print count, '\n', myCentroids, '\n', clustAssing
draw(datMat,myCentroids)
