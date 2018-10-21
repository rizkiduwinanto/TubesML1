from sklearn.datasets import load_iris
from TubesML1 import KMeans, purity_score


def main():
    iris = load_iris()
    x = iris.data
    y = iris.target
    print('Iris data set target')
    print(y)

    kmeans = KMeans(3)
    kmeans.fit(x)
    print('K-means clustering')
    print('Predicted labels:', kmeans.labels)
    print('Centroids:', kmeans.centroids)
    print('Iterations:', kmeans.iters)
    print('Convergent:', kmeans.is_convergent)
    print('Error:', kmeans.error)
    print('Purity score:', purity_score(y, kmeans.labels))


if __name__ == '__main__':
    main()
