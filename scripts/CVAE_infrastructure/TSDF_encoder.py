#!/usr/bin/env python

"""  Add these lines to the launchfile for launching the TSDF_encoder together with the whole ROS 
<!-- TSDF_encoder -->
  <node name="TSDF_encoder" pkg="pointcloud_utils" type="TSDF_encoder.py" output="screen">
    <param name="reconstruct_TSDF" value="false"/>
    <param name="publish_slice" value="false"/> <!-- reconstruct tsdf needs to be true for this to take effect-->
    <param name="model_weight_path" value="$(find pointcloud_utils)/saved_model_weights/latent_dim_100/18-05_11_45/epoch_0020/cp-.ckpt"/>
    <param name="truncation_distance" value="$(arg truncation_distance)" />
    <param name="binarization_threshold" value="0.1" />
  </node>
 """

import sensor_msgs.point_cloud2 as pc2
import ros_numpy as ros_np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
import rospy
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import pickle
import Convolutional_variational_autoencoder as CVAE
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pointcloud_utils.msg import LatSpace
import os
import cv2
from cv_bridge import CvBridge
import math

"""This script defines the TSDF_encoder ROS-node which encodes the TSDF pointclouds in Real-time from the VoxBlox TSDF_server."""

class unordered_pointcloud_to_latent_space():
    def __init__(self,pc_topic="gagarin/tsdf_server/tsdf_pointcloud",lat_topic="/gagarin/pc_latent_space",recon_topic="/reconstructed_pc",
    slice_pub="/pc_slice",voxel_pub="/voxel_pub"):

        #Get robot name
        self.robot_name = rospy.get_param("/robot_name")

        #Import model weight path
        self.arg_filepath = rospy.get_param("~model_weight_path")
        print('__file__:    ', __file__)

        #Load the network
        self.vae = CVAE.VAE()
        self.vae.compile(optimizer=keras.optimizers.Adam())
        self.vae.load_weights(self.arg_filepath)
        self.latent_space_dim = self.vae.latent_dim
        
        #Make subscriber and publisher
        self.pc_sub = rospy.Subscriber(pc_topic,PointCloud2,self.point_cloud_encoder_callback)
        self.lat_pub = rospy.Publisher(lat_topic,LatSpace,queue_size=None)

        self.ctr = 0

        #If reconstruct or not
        if rospy.get_param("~reconstruct_TSDF"):
            self.reconstruct = True
            self.pc_pub = rospy.Publisher(recon_topic,PointCloud2,queue_size=1)
        else:
            self.reconstruct = False

        if rospy.get_param("~publish_slice"):
            self.pub_slice = True
            self.slice_pub = rospy.Publisher(slice_pub,Image,queue_size=1)
        else:
            self.pub_slice = False

        self.bin_thresh = rospy.get_param("~binarization_threshold")


    def point_cloud_encoder_callback(self,pc):

        #Initalize empty array for TSDF to fill
        xyzi = np.zeros((65,65,16))
        
        #Convert from pointcloud2 to numpy array
        arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())

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

        #Threshold the input image
        binarized = np.where(xyzi > self.bin_thresh, 1,0)

        #Format for TF
        tf_input = np.reshape(binarized[:64,:64,:],(64,64,16,1))

        #Do inference
        input_pc = np.array([tf_input])
        latent_space = self.vae.encoder.predict(input_pc)

        #Publish the result
        msg = LatSpace(latent_space=latent_space[0].tolist()[0])
        self.lat_pub.publish(msg)

        #Reconstruct and publish reconstructed pointcloud
        if self.reconstruct == True:
            self.ctr +=1
            #Do inference
            output_image = self.vae.decoder.predict(latent_space[0])[0,:,:,:]

            #Find the scale factor
            scale_factor = np.abs(x_unique[0]-x_unique[1])

            #Convert output image to list of points
            output_image = np.reshape(output_image,(64,64,16))
            cropped_output_image = output_image[:54,:54,:16]
            m,n,l = cropped_output_image.shape
            R,C,T = np.mgrid[:m,:n,:l]*scale_factor

            #Shift points to that the origin matches
            R = R - (scale_factor*27)
            C = C - (scale_factor*27)
            T = T - (scale_factor*8)

            #Create fields
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1)]

            #Create the header
            header = Header()
            header.frame_id = self.robot_name+"/base_link"
            header.stamp = rospy.Time.now()

            #Create the points
            points = np.array([R.ravel(),C.ravel(),T.ravel(),cropped_output_image.ravel()]).reshape(4,-1).T

            #Create the PointCloud2 and filter points outside the visible region of the LiDAR. THIS HAS TO BE CHANGED IF THE LIDAR FOV IS CHANGED FROM ITS 45 DEGREES IN THIS IMPLEMENTATION.
            filtered_points = points[points[:,2]<1.4]
            filtered_points = filtered_points[(np.sqrt(filtered_points[:,0]**2+filtered_points[:,1]**2))<(scale_factor*24)] #Remove all points outside the circle
            filtered_points = filtered_points[(np.sqrt(filtered_points[:,0]**2+filtered_points[:,1]**2))>abs(3*filtered_points[:,2]+0.3)] #Remove all the points in the unobservable cones on top and bottom of the LiDAR
            filtered_points = filtered_points[filtered_points[:,3]<0.3]
            output_pc = pc2.create_cloud(header,fields,filtered_points)

            #Publish pointcloud. This pointcloud can be visualized as a pointcloud in Rviz together with the tsdf_server/occupied_nodes for a visual indication of the encoder performance.
            self.pc_pub.publish(output_pc)
            rospy.loginfo("Published Encoded-Decoded pointcloud")

            #Flip image to match flying direction. Can be visualized as a 2D image in Rviz for evaluation in envs with self-similar obstacles in the z-direction
            flipped_output_image = np.flip(np.flip(cropped_output_image,axis=1),axis=0)
            if self.pub_slice == True:
                image_temp  = CvBridge().cv2_to_imgmsg(flipped_output_image[:,:,8])
                image_temp.header = header
                self.slice_pub.publish(image_temp)
                rospy.loginfo("Published Encoded-Decoded image slice")
                print("Published Encoded-Decoded image slice")



def main():
    rospy.init_node('PC2_encoder')
    pc_saver = unordered_pointcloud_to_latent_space()
    rospy.spin()



if __name__ == '__main__':
    main()