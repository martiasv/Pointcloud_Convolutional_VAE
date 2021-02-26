import numpy as np 
import matplotlib.pyplot as plt
import pickle
import sys

np.set_printoptions(threshold=sys.maxsize)


with open('../pickelled/pointclouds.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pointcloud_array = pickle.load(f)

#print(len(pointcloud_array))
#print(pointcloud_array[0])
#print(np.unique(pointcloud_array[0]))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

print(pointcloud_array[0].shape[0])
print(pointcloud_array[0].shape[1])
print(pointcloud_array[0].shape[2])
#print(pointcloud_array[0,:,1])
#print(pointcloud_array[0,:,2])

pointcloud = pointcloud_array[3]
#print(np.unique(pointcloud))
#print(pointcloud[:,:,10])
for im in range(pointcloud.shape[2]):
    plt.imshow(pointcloud[:,:,im], cmap="gray") 
    plt.show()
    print(np.unique(pointcloud[:,:,im])) 
# for x in range(0,30):#range(pointcloud_array[0].shape[0]):
#     for y in range(0,30):#range(pointcloud_array[0].shape[1]):
#         for z in range(0,10):#range(pointcloud_array[0].shape[2]):
#             ax.scatter(x,y,z,marker='o')

#ax.scatter(list(range(0,pointcloud.shape[0])),list(range(0,pointcloud.shape[1])),list(range(0,pointcloud.shape[2])),marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()