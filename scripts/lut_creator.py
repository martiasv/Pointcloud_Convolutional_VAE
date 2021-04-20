#!/usr/bin/env python

import sensor_msgs.point_cloud2 as pc2
import ros_numpy as ros_np
from sensor_msgs.msg import PointCloud2
import rospy
import numpy as np
import time
from datetime import timedelta
import pickle


class unordered_pointcloud_to_latent_space():
    def __init__(self,pc_topic="/gagarin/tsdf_server/tsdf_pointcloud",lat_topic="/gagarin/pc_latent_space",recon_topic="/gagarin/reconstructed_pc",slice_pub="/gagarin/pc_slice"):

        #Make subscriber and publisher
        self.pc_sub = rospy.Subscriber(pc_topic,PointCloud2,self.point_cloud_encoder_callback)


    def point_cloud_encoder_callback(self,pc):
        #Initalize empty array for TSDF to fill
        xyzi = np.zeros((65,65,32))

        #Convert from pointcloud2 to numpy array
        arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())

        #Convert from decimal position values, to enumerated index values
        x_unique,x_enum = np.unique(arr[:,0],return_inverse= True)
        y_unique,y_enum = np.unique(arr[:,1],return_inverse= True)
        z_unique,z_enum = np.unique(arr[:,2],return_inverse= True)

        x_enumerated = np.arange(len(x_unique)+1).astype(int)
        y_enumerated = np.arange(len(y_unique)+1).astype(int)
        z_enumerated = np.arange(len(z_unique)+1).astype(int)
        # print(x_enumerated)
        # print(y_enumerated)
        # print(z_enumerated)

        #Make lookup tables
        x_lut = [x_unique.tolist(),x_enumerated.tolist()]
        y_lut = [y_unique.tolist(),y_enumerated.tolist()]
        z_lut = [z_unique.tolist(),z_enumerated.tolist()]

        print("X_lut:")
        print(x_lut)
        print("Y_lut:")
        print(y_lut)
        print("Z_lut:")
        print(z_lut)

        with open(f'src/pointcloud_utils/lut/x_lut.pickle', 'wb') as f:
            pickle.dump(x_lut, f, pickle.HIGHEST_PROTOCOL)
            print("Pickled x lut")

        with open(f'src/pointcloud_utils/lut/y_lut.pickle', 'wb') as f:
            pickle.dump(y_lut, f, pickle.HIGHEST_PROTOCOL)
            print("Pickled y lut")

        with open(f'src/pointcloud_utils/lut/z_lut.pickle', 'wb') as f:
            pickle.dump(z_lut, f, pickle.HIGHEST_PROTOCOL)
            print("Pickled z lut")







def main():
    rospy.init_node('PC2_encoder')
    pc_saver = unordered_pointcloud_to_latent_space()
    rospy.spin()



if __name__ == '__main__':
    main()