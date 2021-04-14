#This script shuffles the dataset across batches, so that each batch has the same distribution
##Setup

import pickle
import matplotlib.pyplot as plt
import numpy as np


pointcloud_list = []
num_batches = 4
#names_envs = ['cave','test_tunnel']
#names_envs = ['small_obstacles','large_obstacles','cave','tunnel']

##Load dataset
for i in range(num_batches):
    with open('../pickelled/randomized_3/raw/pointclouds_batch'+str(i+1)+'.pickle', 'rb') as f:
        pointcloud_list.append(np.array(pickle.load(f)))

pointcloud_array = np.reshape(pointcloud_list,(1000*num_batches,65,65,20,1))

print(pointcloud_array.shape)

#Shuffle dataset
np.random.shuffle(pointcloud_array)

print(f'Pickling shuffled pointclouds')
for i in range(num_batches):
    with open(f'../pickelled/randomized_3/shuffled/pointclouds_batch{i+1}.pickle', 'wb') as f:
        pickle.dump(pointcloud_array[i*1000:(i+1)*1000,:,:,:], f, pickle.HIGHEST_PROTOCOL)
    print(f'Batch {i+1}: Pickling complete')
