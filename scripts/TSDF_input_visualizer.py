#!/usr/bin/env python

#   <!-- TSDF_encoder -->
#   <node name="TSDF_encoder" pkg="pointcloud_utils" type="TSDF_encoder.py" output="screen">
#     <param name="reconstruct_TSDF" value="true"/>
#     <param name="publish slice" value="true"/> <!-- reconstruct tsdf needs to be true for this to take effect -->
#     <param name="model_weight_path" value="$(find pointcloud_utils)/saved_model_weights/latent_dim_100/10-04_16_39/epoch_0064/cp-.ckpt"/>
#    </node>

import sensor_msgs.point_cloud2 as pc2
import ros_numpy as ros_np
from sensor_msgs.msg import PointCloud2
import rospy
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import argparse
import os


class unordered_pointcloud_to_latent_space():
    def __init__(self,pc_topic="/gagarin/tsdf_server/tsdf_pointcloud",lat_topic="/gagarin/pc_latent_space",recon_topic="/gagarin/reconstructed_pc",slice_pub="/gagarin/pc_slice",voxel_pub="/gagarin/voxel_pub"):


        self.bin_thresh = 0.1

        #Initalize empty array for TSDF to fill
        xyzi = np.zeros((65,65,16))
        xyzi_ones = np.ones((65,65,16))*(-0.4)

        #Convert from pointcloud2 to numpy array
        with open(f'../pickelled/single_pointcloud_arr.pickle', 'rb') as f:
            arr = pickle.load(f)
            print("Done unpickling the single pointcloud")
        #arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())

        # #Convert from decimal position values, to enumerated index values
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

        #Fill numpy array with the correct intensity value at each index
        xyzi[x_enum,y_enum,z_enum]= arr[:,3]
        xyzi_ones[x_enum,y_enum,z_enum] = arr[:,3]

        plt.imshow(xyzi[:,:,8], cmap="gray") 
        plt.show()

        plt.imshow(xyzi_ones[:,:,8], cmap="gray") 
        plt.show()

        #Threshold the input image
        binarized = np.where(xyzi > self.bin_thresh, 1,0)

        plt.imshow(binarized[:,:,8], cmap="gray") 
        plt.show()




def main():
    pc_saver = unordered_pointcloud_to_latent_space()




if __name__ == '__main__':
    main()