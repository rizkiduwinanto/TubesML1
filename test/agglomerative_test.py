from sklearn.datasets import load_iris
from TubesML1 import Agglomerative, purity_score


def main():
    iris = load_iris()
    x = iris.data
    y = iris.target
    print('Iris data set target')
    print(y)

    agglo_single = Agglomerative(3, 'single')
    agglo_single.fit(x)
    print('Agglomerative clustering (single linkage)')
    print('Predicted labels:', agglo_single.labels)
    print('Predicted clusters:', agglo_single.clusters)
    print('Purity score:', purity_score(y, agglo_single.labels))

    agglo_complete = Agglomerative(3, 'complete')
    agglo_complete.fit(x)
    print('Agglomerative clustering (complete linkage)')
    print('Predicted labels:', agglo_complete.labels)
    print('Predicted clusters:', agglo_complete.clusters)
    print('Purity score:', purity_score(y, agglo_complete.labels))

    agglo_average = Agglomerative(3, 'average')
    agglo_average.fit(x)
    print('Agglomerative clustering (average linkage)')
    print('Predicted labels:', agglo_average.labels)
    print('Predicted clusters:', agglo_average.clusters)
    print('Purity score:', purity_score(y, agglo_average.labels))

    agglo_averagegroup = Agglomerative(3, 'averagegroup')
    agglo_averagegroup.fit(x)
    print('Agglomerative clustering (average group linkage)')
    print('Predicted labels:', agglo_averagegroup.labels)
    print('Predicted clusters:', agglo_averagegroup.clusters)
    print('Purity score:', purity_score(y, agglo_averagegroup.labels))


if __name__ == '__main__':
    main()
