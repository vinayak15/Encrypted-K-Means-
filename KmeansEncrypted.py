import torch
from crypten.mpc.primitives.binary import BinarySharedTensor
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
import crypten.mpc.primitives.circuit
import sys
from crypten.mpc import MPCTensor

import crypten
import crypten.communicator as comm
import crypten.mpc.primitives.converters  as convert
import numpy as np
import matplotlib.pyplot as plt
import crypten.mpc as mpc
import crypten.communicator as comm
import pandas as pd
crypten.init()

torch.set_num_threads(1)
x_enc = crypten.cryptensor([20], ptype=crypten.mpc.arithmetic ,src =0)

x_enc_binary = x_enc.to(crypten.mpc.binary)

x_enc = x_enc.square()

print(x_enc.get_plain_text())
y_enc_binary = crypten.cryptensor([20],src =0)
print(y_enc_binary)
com = x_enc.gt(y_enc_binary)
print(com)
print(com.get_plain_text())

dataset=pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3, 4]].values
dataset.describe()

#print(X)

max_epoch = 5

k = 5

if k > len(X):
    print("K means not possible as cluster is greater than  dataset")
    sys.exit()

enc_dataset = []
for x in X:
    tensor = crypten.cryptensor(x)
    enc_dataset.append((tensor, -1))

clusters = [dict() for x in range(k)]
for x in range(k):
    clusters[x]['coordinate'] = enc_dataset[x][0]  # initiallize the clusters with random data points
    clusters[x]['elements'] = []
for iteration in range(max_epoch):  # around 100 epochs for convergence because
    # of randon initiialization
    for point_index in range(len(enc_dataset)):
        distance = []
        for index, cluster in enumerate(clusters):
            #print("squared")
            # print("vinayak")
            # print(point[0].get_plain_text())
            # print(cluster['coordinate'].get_plain_text())
            substracted_tensor = enc_dataset[point_index][0].sub(cluster['coordinate'])
            # print(substracted_tensor.get_plain_text())
            squared = substracted_tensor.square()
            #print(squared)
            squared = squared.sum()
            #print(squared.get_plain_text())

            #squared = convert._A2B(squared)  # Convert tensor to Binary shared tensor for comparison
            distance.append((squared, index))

        smallest,smallest_index = distance[0][0], distance[0][1]
        # print(distance[0][0].get_plain_text())
        for i in range(len(distance)):
            # print(" ")
            # print("loop start")
            # print("distance unencrpyted " + str(distance[i][0].get_plain_text()))
            #
            # print("smalleset unencrypted"+ str(smallest.get_plain_text()))
            # print("distance encrypted"+ str(distance[i][0]))
            # #print(smallest)
            # print("smaleest encrypted"+ str(smallest))
            comparison = smallest.lt(distance[i][0])
            #comparison=comparison.get_plain_text()
            #print("comparison "+ str(comparison))

            if comparison.get_plain_text() <= 0:
                smallest_index = distance[i][1]
            #smallest = distance[i][0]

            one = crypten.cryptensor(torch.Tensor([1]))

            result1 = comparison.mul(smallest)
            #print("result1"+ str(result1.get_plain_text()))
            result2 = one.sub(comparison)
            #print(result2)
            result2 = result2.mul(distance[i][0])
            #print("result2"+ str(result2))
            smallest = result1.add(result2)
            #print("smallles"+ str(smallest.get_plain_text()))
            #print(smallest.get_plain_text())
            # smallest_index = ArithmeticSharedTensor(smallest_index)
            # result1 = comparison.mul(smallest_index)
            # result2 = one.sub(comparison)
            # result2 = result2.mul(distance[i][1])
            # smallest_index = result1.add(result2)
            # smallest_index = smallest_index.get_plain_text()
        # print("viniauak")
        # print(smallest.get_plain_text())
        # print(smallest_index)
        #smallest = convert._B2A(smallest)  # convert tensor back to Arthimatic shared tensor
        if enc_dataset[point_index][1] != -1:
            clusters[enc_dataset[point_index][1]]['elements'].remove(enc_dataset[point_index][0])
        clusters[int(smallest_index)]['elements'].append(enc_dataset[point_index][0])
        enc_dataset[point_index] =(enc_dataset[point_index][0], int(smallest_index))
    # for x in range(k):
    #     print('clusters  = '+ str(x))
    #     print(clusters[x]['coordinate'].get_plain_text())
    #     [print(y.get_plain_text()) for y in clusters[x]['elements']]
    for cluster in clusters:
        x = crypten.cryptensor(torch.Tensor([0, 0]))
        for data in cluster['elements']:
            x = x.add(data)
        if len(cluster['elements']) !=0:
            x = x.div(len(cluster['elements']))
            cluster['coordinate'] = x

un_enctyped_clusters = [dict() for x in range(k)]
for x in range(k):
    un_enctyped_clusters[x]['coordinate'] = clusters[x]['coordinate'].get_plain_text()
    un_enctyped_clusters[x]['elements'] = [y.get_plain_text() for y in clusters[x]['elements']]

color_total = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'white']
labels_total = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5','centroid']

# print("un_encrpyted clusters")
# print(un_enctyped_clusters)
x_array_centroid = []
y_array_centroid = []
color_centroid = []
for index,x in enumerate(un_enctyped_clusters):
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
    plt.scatter(x_array,y_array,s=100,label=labels_total[index],color=color)
plt.scatter(x_array_centroid,y_array_centroid,s=100,label=labels_total[5],color=color_centroid)

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

