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
from sensor_msgs.msg import PointField
from sensor_msgs.msg import Image
import rospy
import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import pickle
import Convolutional_variational_autoencoder as CVAE
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Header
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pointcloud_utils.msg import LatSpace
import os
import cv2
from cv_bridge import CvBridge

class unordered_pointcloud_to_latent_space():
    def __init__(self,pc_topic="/gagarin/tsdf_server/tsdf_pointcloud",lat_topic="/gagarin/pc_latent_space",recon_topic="/gagarin/reconstructed_pc",slice_pub="/gagarin/pc_slice"):

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



    def point_cloud_encoder_callback(self,pc):
        #Initalize empty array for TSDF to fill
        xyzi = np.zeros((65,65,24))

        #Convert from pointcloud2 to numpy array
        arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())

        #Convert from decimal position values, to enumerated index values
        x_unique,x_enum = np.unique(arr[:,0],return_inverse= True)
        _,y_enum = np.unique(arr[:,1],return_inverse= True)
        _,z_enum = np.unique(arr[:,2],return_inverse= True)

        #Fill numpy array with the correct intensity value at each index
        xyzi[x_enum,y_enum,z_enum]= arr[:,3]

        #Format for TF
        tf_input = np.reshape(xyzi[:64,:64,:],(64,64,24,1))

        #Do inference
        input_pc = np.array([tf_input])
        #latent_space = self.vae.encoder.predict(input_pc)[0].tolist()[0]
        latent_space = self.vae.encoder.predict(input_pc)

        #Publish the result
        msg = LatSpace(latent_space=latent_space[0].tolist()[0])
        self.lat_pub.publish(msg)

        #Reconstruct and publish reconstructed pointcloud
        if self.reconstruct == True:
            #Do inference
            output_image = self.vae.decoder.predict(latent_space[0])[0,:,:,:]

            #Find the scale factor
            scale_factor = np.abs(x_unique[0]-x_unique[1])

            #Convert output image to list of points
            output_image = np.reshape(output_image,(64,64,24))
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
            header.frame_id = "gagarin/base_link"
            header.stamp = rospy.Time.now()

            #Create the points
            points = np.array([R.ravel(),C.ravel(),T.ravel(),cropped_output_image.ravel()]).reshape(4,-1).T

            #Create the PointCloud2
            output_pc = pc2.create_cloud(header,fields,points)

            #Publish pointcloud
            self.pc_pub.publish(output_pc)
            rospy.loginfo("Published Encoded-Decoded pointcloud")

            if self.pub_slice == True:
                image_temp  = CvBridge().cv2_to_imgmsg(cropped_output_image[:,:,8])
                image_temp.header = header
                self.slice_pub.publish(image_temp)
                rospy.loginfo("Published Encoded-Decoded image slice")





def main():
    rospy.init_node('PC2_encoder')
    pc_saver = unordered_pointcloud_to_latent_space()
    rospy.spin()



if __name__ == '__main__':
    main()