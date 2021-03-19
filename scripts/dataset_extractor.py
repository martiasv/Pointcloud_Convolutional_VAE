#This script extracts data from the dataset
##Setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import matplotlib.pyplot as plt
import numpy as np
from random import randrange

pointcloud_list = []
num_batches = 4
names_envs = ['large_obstacle_tunnel','large_obstacles','cave','tunnel']

##Load dataset
for name in names_envs:
    with open('../pickelled/' +name + '/shuffled/pointclouds_batch1.pickle', 'rb') as f:
        pointcloud_list.append(np.array(pickle.load(f)))

pointcloud_array = np.reshape(pointcloud_list,(1000*num_batches,65,65,20,1))

print(pointcloud_array.shape)

#Draw random indices numbers
test_set = []
rand_nrs_1 = [randrange(0, 1000) for x in range(5)]
rand_nrs_2 = [randrange(1001, 2000) for x in range(5)]
rand_nrs_3 = [randrange(2001, 3000) for x in range(5)]
rand_nrs_4 = [randrange(3001, 4000) for x in range(5)]
rand_nr_list = []
rand_nr_list.append(rand_nrs_1)
rand_nr_list.append(rand_nrs_2)
rand_nr_list.append(rand_nrs_3)
rand_nr_list.append(rand_nrs_4)
rand_nr_list = np.array(rand_nr_list)
rand_nr_list = rand_nr_list.flatten()
print(rand_nr_list.shape)
print(rand_nr_list)

#Inspection of input data
#for im in range(pointcloud_array.shape[2]-1):
for i in rand_nr_list:
    plt.imshow(pointcloud_array[i,:64,:64,8,0], cmap="gray") 
    plt.show()

print(f'Pickling new dataset pointclouds')
for i in range(num_batches):
    with open(f'../pickelled/test_set_4envs/test_set_4envs.pickle', 'wb') as f:
        pickle.dump(pointcloud_array[rand_nr_list,:,:,:], f, pickle.HIGHEST_PROTOCOL)
    print(f'Batch {i+1}: Pickling complete')
