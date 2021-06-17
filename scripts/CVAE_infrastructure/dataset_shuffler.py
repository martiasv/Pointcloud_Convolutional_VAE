#This script shuffles the dataset across batches, so that each batch has the same distribution of pointcloud examples. 
#Run this outside of the ROS framework, call "python dataset_shuffler.py"

import pickle
import matplotlib.pyplot as plt
import numpy as np


pointcloud_list = []
num_batches = 1

##Load dataset
for i in range(num_batches):
    with open('../pickelled/test_corridor_yawless_spawn/raw/pointclouds_batch'+str(i+1)+'.pickle', 'rb') as f:
        pointcloud_list.append(np.array(pickle.load(f)))

pointcloud_array = np.reshape(pointcloud_list,(1000*num_batches,65,65,16,1))

print(pointcloud_array.shape)

#Shuffle dataset
np.random.shuffle(pointcloud_array)

print(f'Pickling shuffled pointclouds')
for i in range(num_batches):
    with open(f'../pickelled/test_corridor_yawless_spawn/shuffled/pointclouds_batch{i+1}.pickle', 'wb') as f:
        pickle.dump(pointcloud_array[i*1000:(i+1)*1000,:,:,:], f, pickle.HIGHEST_PROTOCOL)
    print(f'Batch {i+1}: Pickling complete')
