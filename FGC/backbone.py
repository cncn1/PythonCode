from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances_argmin
import numpy as np
import warnings
from sklearn.cluster import AgglomerativeClustering
from time import time

warnings.filterwarnings("ignore")


def backbone(data, kind, name, n, n_digits, labels, seed2=1, number=100):
    Initdata = data
    t0 = time()
    while (number != 0):
        lastnumber = -100
        number = 100
        seed2 = seed2 + 1
        lastsize = -1
        while (lastnumber != number):
            lastnumber = number
            # print lastnumber,lastsize
            number = 0
            estimeter = []
            k_means_cluster_centers = []
            k_means_labels = []
            order = []
            index = 0
            seed = 0
            while (index < n):
                e = KMeans(init=kind, n_clusters=n_digits, random_state=seed2 * seed * (1 + number)).fit(data);
                if (np.shape(np.unique(pairwise_distances_argmin(data, e.cluster_centers_)))[0] ==
                        np.shape(np.unique(labels))[0]):
                    # print np.shape(np.unique(pairwise_distances_argmin(data,e.cluster_centers_)))[0],np.unique(pairwise_distances_argmin(data,e.cluster_centers_))
                    estimeter.append(e)
                    k_means_cluster_centers.append(np.sort(e.cluster_centers_, axis=0))
                    k_means_labels.append(pairwise_distances_argmin(data, e.cluster_centers_))
                    index = index + 1
                seed = seed + 1
            for i in range(n - 1):
                order.append(pairwise_distances_argmin(k_means_cluster_centers[0], k_means_cluster_centers[i + 1]))
            different = (k_means_labels[0] == 11)
            # print different
            for i in range(n - 1):
                for k in range(n_digits):
                    different += ((k_means_labels[0] == k) != (k_means_labels[i + 1] == order[i][k]))
            for i in different:
                if i == True:
                    number = number + 1
            # print k_means_labels,order,different,number
            tempdata = []
            for i in range(len(data)):
                if (different[i]):
                    tempdata.append(data[i])
            for i in range(n_digits):
                #  print data.tolist()
                reducePoints = []
                reducePointsIndex = filter(lambda x: different[x] == False and k_means_labels[0][x] == i,
                                           range(len(data)))
                for j in reducePointsIndex:
                    reducePoints.append(data[j])
                Core = np.mean(reducePoints, axis=0)
                #   print Core
                try:
                    np.shape(Core)[0]
                    tempdata.append(Core)
                except:
                    reducePoints = []
                    #   print Core,reducePointsIndex,i,np.unique(k_means_labels[0])
                # print np.shape(data)
            data = tempdata

            # print np.shape(tempdata),np.shape(data),number,lastnumber
            if (lastnumber == number):
                break
            if (lastsize == np.shape(tempdata)[0]):
                break
            lastsize = np.shape(tempdata)[0]
            if (np.shape(tempdata)[0] == np.shape(np.unique(k_means_labels[0]))[0]):
                number = 0
                break
    print('% 9s           %.2fs            %.3f         %.3f      %.3f      %.3f      %.3f        %.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.completeness_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.v_measure_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.adjusted_rand_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.adjusted_mutual_info_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.silhouette_score(Initdata, pairwise_distances_argmin(Initdata, tempdata),
                                      metric='euclidean',
                                      sample_size=300)))


def modifiedbackbone(data, kind, name, n, n_digits, labels, seed2=1, number=100):
    sample_max = np.shape(data)[1]
    Initdata = data
    t0 = time()
    while (number != 0):
        lastnumber = -100
        number = 100
        seed2 = seed2 + 1
        lastsize = -1
        while (lastnumber != number):
            lastnumber = number
            # print lastnumber,lastsize
            number = 0
            estimeter = []
            k_means_cluster_centers = []
            k_means_labels = []
            order = []
            index = 0
            seed = 0
            while (index < n):
                e = KMeans(init=kind, n_clusters=n_digits, random_state=seed2 * seed * (1 + number)).fit(data);
                if (np.shape(np.unique(pairwise_distances_argmin(data, e.cluster_centers_)))[0] ==
                        np.shape(np.unique(labels))[0]):
                    # print np.shape(np.unique(pairwise_distances_argmin(data,e.cluster_centers_)))[0],np.unique(pairwise_distances_argmin(data,e.cluster_centers_))
                    estimeter.append(e)
                    k_means_cluster_centers.append(np.sort(e.cluster_centers_, axis=0))
                    k_means_labels.append(pairwise_distances_argmin(data, e.cluster_centers_))
                    index = index + 1
                seed = seed + 1
            for i in range(n - 1):
                order.append(pairwise_distances_argmin(k_means_cluster_centers[0], k_means_cluster_centers[i + 1]))
            different = (k_means_labels[0] == 11)
            # print different
            for i in range(n - 1):
                for k in range(n_digits):
                    different += ((k_means_labels[0] == k) != (k_means_labels[i + 1] == order[i][k]))
            for i in different:
                if i == True:
                    number = number + 1
            # print k_means_labels,order,different,number
            tempdata = []

            for i in range(n_digits):
                #  print data.tolist()
                reducePoints = []
                reducePointsIndex = filter(lambda x: different[x] == False and k_means_labels[0][x] == i,
                                           range(len(data)))
                for j in reducePointsIndex:
                    reducePoints.append(data[j])
                Core = np.mean(reducePoints, axis=0)
                #   print Core
                try:
                    for i in range(len(reducePoints)):
                        tempdata.append(Core)
                except:
                    reducePoints = []
                    #   print Core,reducePointsIndex,i,np.unique(k_means_labels[0])
                    # print np.shape(data)
            # lastsize=np.shape(data)[0]
            for i in range(len(data)):
                if (different[i]):
                    tempdata.append(data[i])
            data = tempdata

            if (lastnumber == number):
                break
            if (lastsize == np.shape(np.unique(data))):
                break
                # print lastsize,np.shape(np.unique(data))
            lastsize = np.shape(np.unique(data))
            # print np.shape(data)[0],n_digits
            if (np.shape(np.unique(data))[0] <= n_digits * sample_max):
                number = 0
                n = 0
                break
    print('% 9s           %.2fs         %.3f       %.3f      %.3f      %.3f      %.3f        %.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.completeness_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.v_measure_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.adjusted_rand_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.adjusted_mutual_info_score(labels, pairwise_distances_argmin(Initdata, tempdata)),
             metrics.silhouette_score(Initdata, pairwise_distances_argmin(Initdata, tempdata),
                                      metric='euclidean',
                                      sample_size=300)))
