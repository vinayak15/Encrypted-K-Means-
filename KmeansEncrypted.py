import sys

import logging
import crypten
import matplotlib.pyplot as plt
import pandas as pd
import torch
from crypten.mpc import MPCTensor


def kmeanspp(enc_dataset, K):
    clusters = [dict() for x in range(K)]
    clusters[0]['coordinate'] = enc_dataset[0][0]
    clusters[0]['elements'] = []

    for k in range(1, K):
        distance_list = []
        for point in enc_dataset:
            minimum_centroid_distance = []
            for index in range(k):
                diff = clusters[index]['coordinate'].sub(point[0])
                diff = diff.square()
                x = diff.sum()
                minimum_centroid_distance.append(x)
            minimum_centroid_distance = MPCTensor.stack(minimum_centroid_distance)

            indices = minimum_centroid_distance.min()

            distance_list.append(indices)

        distance_list = MPCTensor.stack(distance_list)
        #print(distance_list.get_plain_text())
        maximum_distance_index = distance_list.argmax().reveal()
        #print(crypten.communicator.get().get_rank())
        # if crypten.communicator.get().get_rank() == 0:
        #     print(maximum_distance_index)
        #print(maximum_distance_index)
        for index, value in enumerate(maximum_distance_index):
            if value == 1:
                smallest_index = index
        clusters[k]['coordinate'] = enc_dataset[smallest_index][0]
        clusters[k]['elements'] =[]
    return clusters

def print_clusters(clusters, k):
    print("Intial cluster locations")
    # for i in range(k):
    #     print(clusters[i]['coordinate'].get_plain_text() ," ")
    #     print(clusters[i]['coordinate'] ," ")
    #
    #     print("length of this cluseter"+ str(len(clusters[i]['elements'])))
    clusters = decrypt_clusters(clusters,k)
    verify_clusters(clusters)


def train_kmeans(enc_dataset,max_epoch, k):
    clusters = [dict() for x in range(k)]
    # for x in range(k):
    #     clusters[x]['coordinate'] = enc_dataset[x][0]  # initiallize the clusters with random data points
    #     clusters[x]['elements'] = []
    clusters = kmeanspp(enc_dataset, k)  # kmeans ++ initialization so that cluster are corectly formed
    print_clusters(clusters, k)

    for iteration in range(max_epoch):  # around 100 epochs for convergence because
        for point_index in range(len(enc_dataset)):
            distance = []
            for index, cluster in enumerate(clusters):
                substracted_tensor = enc_dataset[point_index][0].sub(cluster['coordinate'])
                squared = substracted_tensor.square()
                squared = squared.sum()
                distance.append(squared)
            distance_in_tensor = MPCTensor.stack(distance)
            #print(distance_in_tensor.get_plain_text())
            indices = distance_in_tensor.argmin().reveal()  # this just converts Artihmatic tensor to tnesor with encrpyted values intact,
            #print(indices)                                     # no decrpytion happen here
            for index, value in enumerate(indices):
                if value == 1:
                    smallest_index = index
                    break
            if enc_dataset[point_index][1] != -1:
                clusters[enc_dataset[point_index][1]]['elements'].remove(enc_dataset[point_index][0])
            clusters[smallest_index]['elements'].append(enc_dataset[point_index][0])
            enc_dataset[point_index] = (enc_dataset[point_index][0], smallest_index)
        for cluster in clusters:
            x = crypten.cryptensor([0.0, 0.0])

            for data in cluster['elements']:
                x = x.add(data)
            if len(cluster['elements']) != 0:
                x = x.div(len(cluster['elements']))
                x = x.get_plain_text()              #this is done because division operation for floating number is not correct in crypten
                a=int(x[0])                         # will be changed in future updated, so we are converting to int and then using same
                b=int(x[1])                         # if diviison worked properly we would not need to decrypt plain value here
                x= crypten.cryptensor([a,b])

                x=crypten.cryptensor(x)
                cluster['coordinate'] = x
        #print_clusters(clusters,k)
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


def run_mpc_kmeans(epochs=5, input_path=None, k=2, skip_plaintext=False):
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
    # if crypten.communicator.get().get_rank() == 0:
    logging.info("==================")
    logging.info("Decrypting Clusters ")
    logging.info("==================")
    decrypted_clusters = decrypt_clusters(clusters, k)


    # if crypten.communicator.get().get_rank() == 0:
    logging.info("==================")
    logging.info("Printing  Clusters ")
    logging.info("==================")
    verify_clusters(decrypted_clusters)
    #verify_clusters(clusters)

