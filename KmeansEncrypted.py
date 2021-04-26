import sys
import random as rd

import logging
import crypten
import crypten.mpc.primitives.circuit
import matplotlib.pyplot as plt
import pandas as pd
import torch
from crypten.mpc import MPCTensor


def kmeanspp(enc_dataset, K):
    clusters = [dict() for x in range(K)]
    i = rd.randint(0, len(enc_dataset))

    clusters[0]['coordinate'] = enc_dataset[i][0]
    clusters[0]['elements'] = []
    for k in range(1, K):
        distance_list = []
        for point in enc_dataset:
            minimum_centroid_distance = []
            #print(k)
            for index in range(k):
                #print(point[0].get_plain_text())
                diff = clusters[index]['coordinate'].sub(point[0])
                diff = diff.square()
                x = diff.sum()
                minimum_centroid_distance.append(x)
            minimum_centroid_distance = MPCTensor.stack(minimum_centroid_distance)
            #print("vinayak")
            #print(minimum_centroid_distance.get_plain_text())
            indices = minimum_centroid_distance.min()
            #print(indices.get_plain_text())
            distance_list.append(indices)
        distance_list = MPCTensor.stack(distance_list)
        maximum_distance_index = distance_list.argmax().reveal()
        smallest_index =-1
        #print(crypten.communicator.get().get_rank())
        # if crypten.communicator.get().get_rank() == 0:
        #     print(maximum_distance_index)
        for index, value in enumerate(maximum_distance_index):
            if value == 1:
                smallest_index = index
        clusters[k]['coordinate'] = enc_dataset[smallest_index][0]
        clusters[k]['elements'] =[]

    return clusters


def train_kmeans(enc_dataset, max_epoch, k):
    # clusters = [dict() for x in range(k)]
    # for x in range(k):
    #     clusters[x]['coordinate'] = enc_dataset[x][0]  # initiallize the clusters with random data points
    #     clusters[x]['elements'] = []

    clusters = kmeanspp(enc_dataset, k)  # kmeans ++ initialization so that cluster are corectly formed
    # if crypten.communicator.get().get_rank() == 0:
    #     print(clusters)
    for iteration in range(max_epoch):  # around 100 epochs for convergence because
        # of randon initiialization
        for point_index in range(len(enc_dataset)):
            distance = []
            for index, cluster in enumerate(clusters):
                substracted_tensor = enc_dataset[point_index][0].sub(cluster['coordinate'])
                squared = substracted_tensor.square()
                squared = squared.sum()

                distance.append(squared)

            distance_in_tensor = MPCTensor.stack(distance)
            smallest_index = 0
            indices = distance_in_tensor.argmin().reveal()  # this just converts Artihmatic tensor to tnesor with encrpyted values intact,
            # print(indices)                                     # no decrpytion happen here
            for index, value in enumerate(indices):
                if value == 1:
                    smallest_index = index

            if enc_dataset[point_index][1] != -1:
                clusters[enc_dataset[point_index][1]]['elements'].remove(enc_dataset[point_index][0])
            clusters[int(smallest_index)]['elements'].append(enc_dataset[point_index][0])
            enc_dataset[point_index] = (enc_dataset[point_index][0], int(smallest_index))
        for cluster in clusters:
            x = crypten.cryptensor(torch.Tensor([0, 0]))
            for data in cluster['elements']:
                x = x.add(data)
            if len(cluster['elements']) != 0:
                x = x.div(len(cluster['elements']))
                cluster['coordinate'] = x

    return clusters


def decrypt_clusters(clusters, k):
    un_enctyped_clusters = [dict() for x in range(k)]
    for x in range(k):
        un_enctyped_clusters[x]['coordinate'] = clusters[x]['coordinate'].get_plain_text()
        un_enctyped_clusters[x]['elements'] = [y.get_plain_text() for y in clusters[x]['elements']]
    return un_enctyped_clusters


def verify_clusters(un_enctyped_clusters):
    color_total = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'white']
    labels_total = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'centroid']

    x_array_centroid = []
    y_array_centroid = []
    color_centroid = []
    for index, x in enumerate(un_enctyped_clusters):
        x_array = []
        y_array = []
        color = []

        for y in x['elements']:
            x_array.append(y[0])
            y_array.append(y[1])
            color.append(color_total[index])
        x_array_centroid.append(x['coordinate'][0])
        y_array_centroid.append(x['coordinate'][1])
        color_centroid.append(color_total[6])
        plt.scatter(x_array, y_array, s=100, label=labels_total[index], color=color)
    plt.scatter(x_array_centroid, y_array_centroid, s=100, label=labels_total[5], color=color_centroid)

    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()


def print_dataset(X):
    plt.scatter(X[:, 0], X[:, 1], c='black', label='unclustered data')
    plt.xlabel('Income')
    plt.ylabel('Number of transactions')
    plt.legend()
    plt.title('Plot of data points')
    plt.show()


def run_mpc_kmeans(epochs=5, input_path=None, k=2, skip_plaintext=False, rank='0'):
    crypten.init()
    torch.manual_seed(1)
    dataset = pd.read_csv(input_path)
    X = dataset.iloc[:, [3, 4]].values
    dataset.describe()

    #print_dataset(X)

    if k > len(X):
        print("K means not possible as cluster is greater than  dataset")
        sys.exit()

    enc_dataset = []

    for x in X:
        tensor = crypten.cryptensor(x)
        enc_dataset.append((tensor, -1))


    logging.info("==================")
    logging.info("CrypTen K Means  Training")
    logging.info("==================")
    clusters = train_kmeans(enc_dataset, epochs, k)
    if crypten.communicator.get().get_rank() == 0:
        logging.info("==================")
        logging.info("Decrypting Clusters ")
        logging.info("==================")
    decrypted_clusters = decrypt_clusters(clusters, k)


    if crypten.communicator.get().get_rank() == 0:
        logging.info("==================")
        logging.info("Printing  Clusters ")
        logging.info("==================")
        verify_clusters(decrypted_clusters)
