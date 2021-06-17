#!/usr/bin/env python

#import tensorflow as tf 
import sensor_msgs.point_cloud2 as pc2
import ros_numpy as ros_np
from sensor_msgs.msg import PointCloud2
import rospy
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import pickle

"""This script collects PoincCloud2 TSDF messages from the VoxBlox TSDF server package, converts them to numpy arrays and pickles them. """

class unordered_pointcloud_to_3Darray_dataset():
    def __init__(self,pc_topic="/tsdf_server/tsdf_pointcloud"):
        self.pc_sub = rospy.Subscriber(pc_topic,PointCloud2,self.point_cloud_callback)
        self.XYZI_pointclouds = []
        self.count = 0
        self.bin_thresh = 0.1

    def point_cloud_callback(self,pc):

        #Create placeholder tensor
        xyzi = np.zeros((65,65,16))

        #Convert the points from the PointCloud2 message to a list of coordinates with their corresponding intensity values
        arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())

        #Find all the unique values in the list, and convert the coordinate position values in the pointcloud list to indexes for the pointcloud tensor
        x_unique,x_enum = np.unique(arr[:,0],return_inverse= True)
        y_unique,y_enum = np.unique(arr[:,1],return_inverse= True)
        z_unique,z_enum = np.unique(arr[:,2],return_inverse= True)

        #Count the number of occurences smaller than 0 to shift values for correct values when TSDF map is initialized
        x_smaller = (x_unique<0).sum()
        y_smaller = (y_unique<0).sum()
        z_smaller = (z_unique<0).sum()

        #Shift values based on number of occurences
        x_enum = x_enum + (27-x_smaller)
        y_enum = y_enum + (27-y_smaller)
        z_enum = z_enum + (8-z_smaller)

        #Broadcast values into the new tensor. 
        xyzi[x_enum,y_enum,z_enum]= arr[:,3]

        #Threshold the input image
        xyzi = np.where(xyzi > self.bin_thresh, 1,0)

        #Reduce size of pointcloud by decreasing precision to a uint8 value instead of float64
        xyzi = np.array(xyzi,dtype='B')

        #Append the processed pointcloud to the array
        self.count +=1
        self.XYZI_pointclouds.append(xyzi)
        print(f'{self.count} pointclouds')

        #For every 1000 iteration, save pickle to file
        if len(self.XYZI_pointclouds)%1000==0:
            with open(f'src/pointcloud_utils/pickelled/validation_env/pointclouds_batch{self.count//1000}.pickle', 'wb') as f:
                pickle.dump(self.XYZI_pointclouds, f, pickle.HIGHEST_PROTOCOL)
            print(f'Batch {self.count//1000}: Pickling complete')
            self.XYZI_pointclouds = []



def main():

    #Make the node and run it in parallel. It will then save pointclouds sent to the pc_topic defined earlier
    rospy.init_node('PC2_subscriber')
    pc_saver = unordered_pointcloud_to_3Darray_dataset()
    rospy.spin()




if __name__ == '__main__':
    main()