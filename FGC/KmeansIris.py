from time import time
import numpy as np
import matplotlib.pyplot as plt
import backbone as bb
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

np.random.seed(42)

digits = load_iris()
data = digits.data

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300


def bench(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s           %.2fs            %.3f         %.3f      %.3f      %.3f      %.3f        %.3f'
          % (name, (time() - t0),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


def main():
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))

    print(100 * '_')
    print('% 9s' % 'init'
                   '    time            homo         compl       v-meas        ARI        AMI        silhouette')

    bench(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
          name="k-means++K-means", data=data)

    bench(KMeans(init='random', n_clusters=n_digits, n_init=10),
          name="randomK-means", data=data)

    pca = PCA(n_components=n_digits).fit(data)
    bench(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
          name="PCA-basedK-means",
          data=data)
    bench(AgglomerativeClustering(linkage='ward', n_clusters=n_digits),
          name="AgglomerativeWard",
          data=data)
    bench(AgglomerativeClustering(linkage='complete', n_clusters=n_digits),
          name="AgglomerativeComplete",
          data=data)
    bench(AgglomerativeClustering(linkage='average', n_clusters=n_digits),
          name="AgglomerativeAverage",
          data=data)
    bb.modifiedbackbone(data, 'k-means++', "ModifiedKMeans++backbone", 5, n_digits, labels)
    bb.backbone(data, 'k-means++', "KMeans++backbone", 5, n_digits, labels)
    bb.backbone(data, 'random', "KMeansRandombackbone", 2, n_digits, labels)
    bb.backbone(data, pca.components_, "KMeansPCAbackbone", 2, n_digits, labels)

    print(100 * '_')
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    main()
