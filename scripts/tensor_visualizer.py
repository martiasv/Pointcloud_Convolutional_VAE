import numpy as np 
import matplotlib.pyplot as plt
import pickle
import sys

np.set_printoptions(threshold=sys.maxsize)


with open('../pickelled/pointclouds.pickle', 'rb') as f:
    pointcloud_array = pickle.load(f)


print(pointcloud_array[0].shape[0])
print(pointcloud_array[0].shape[1])
print(pointcloud_array[0].shape[2])

pointcloud = pointcloud_array[3]

for im in range(pointcloud.shape[2]):
    plt.imshow(pointcloud[:,:,im], cmap="gray") 
    plt.show()
    print(np.unique(pointcloud[:,:,im])) 
