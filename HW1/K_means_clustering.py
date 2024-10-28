import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data(file_name):
    dataframe = pd.read_csv(file_name)
    # print(dataframe)
    data = dataframe.values
    # print(data)
    return data

def find_best_k(data):
    inertias = []
    k_range = [k for k in range(1,15+1)]                    # initialize xlabel and we will apply k-means 15 times

    for k in k_range:
        kmeans = KMeans(n_clusters = k, n_init = 10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(k_range, inertias, 'x-')
    plt.xlabel('number of clusters(k)')
    plt.ylabel('Inertia')
    plt.title('inertias and k')
    plt.show()

def reapply_kmeans(data, k = 4):
    kmeans = KMeans(n_clusters = k, n_init = 10)
    kmeans.fit(data)
    labels = kmeans.labels_
    print(labels)
    for i in range(k):
        number = len([y for y in labels if y == i])
        print(f' observation in class {i} is {number}')
    print(f'inertia when k = {k} is: {kmeans.inertia_}')
    visualize_data(data, labels, k)

def visualize_data(data, labels, k):
    for i in range(k):
        idx = [j for j in range(len(data)) if labels[j] == i]
        each_cluster = data[idx]
        # print(each_cluster)
        plt.scatter(each_cluster[:,0], each_cluster[:,1])
    plt.legend([f'class {i}' for i in range(1,k+1)])
    plt.title(f'K-means Clustering with {k} Clusters')
    plt.xlabel('first variable of the data')
    plt.ylabel('second variable of the data')
    plt.show()








if __name__ == '__main__':
    data = load_data(file_name = 'clust_data.csv')
    find_best_k(data)

    # from the graph we can see that the elbow point appears when k = 4

    reapply_kmeans(data, k = 4)