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
import Minimal_convolutional_variational_autoencoder as MCVAE
from std_msgs.msg import Float32MultiArray, Float32
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
    def __init__(self,pc_topic="/tsdf_server/tsdf_pointcloud",lat_topic="/delta/pc_latent_space",
            recon_topic="/reconstructed_pc",slice_topic="/pc_slice", dist_topic="/delta/shortest_distance", 
            bin_entropy_topic="/delta/bin_entropy",voxel_pub="/delta/voxel_pub"):

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
        
        if rospy.get_param("~publish_distance"):
            self.publish_distance = True
            self.dist_pub = rospy.Publisher(dist_topic,Float32,queue_size=1)
        else:
            self.publish_distance = False

        if rospy.get_param("~publish_bin_entropy"):
            self.publish_bin_entropy = True
            self.bin_entropy_pub = rospy.Publisher(bin_entropy_topic,Float32,queue_size=1)
        else:
            self.publish_bin_entropy = False


        if rospy.get_param("~publish_distance"):
            self.publish_distance = True
            self.dist_pub = rospy.Publisher(dist_topic,Float32,queue_size=1)
        else:
            self.publish_distance = False
        
        if rospy.get_param("~publish_bin_entropy"):
            self.publish_bin_entropy = True
            self.bin_entropy_pub = rospy.Publisher(bin_entropy_topic,Float32,queue_size=1)
        else:
            self.publish_bin_entropy = False

        if rospy.get_param("~publish_slice"):
            self.pub_slice = True
            self.slice_pub = rospy.Publisher(slice_topic,Image,queue_size=1)
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

        if (self.publish_distance == True):
            shortest_distance = self.find_closest_distance(arr)
            dist_msg = Float32(shortest_distance)
            self.dist_pub.publish(dist_msg)

        #Reconstruct and publish reconstructed pointcloud
        if self.reconstruct == True:
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

            if(self.publish_bin_entropy == True):
                true_tsdf = binarized
                binary_crossentropy = self.find_binary_crossentropy(true_tsdf, output_image) #Extracting a 3D neighborhood around the drone
                print("Binary_crossentropy: ", binary_crossentropy)
                bin_entropy_msg = Float32(binary_crossentropy)
                self.bin_entropy_pub.publish(bin_entropy_msg)
            
            #Create fields
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1)]

            #Create the header
            header = Header()
            header.frame_id = "delta/base_link"
            header.stamp = rospy.Time.now()

            #Create the points
            points = np.array([R.ravel(),C.ravel(),T.ravel(),cropped_output_image.ravel()]).reshape(4,-1).T

            #Create the PointCloud2
            output_pc = pc2.create_cloud(header,fields,points)

            #Publish pointcloud
            self.pc_pub.publish(output_pc)
            rospy.loginfo("Published Encoded-Decoded pointcloud")

            if self.pub_slice == True:
                #Flip image to match flying direction
                flipped_output_image = np.flip(np.flip(cropped_output_image[:,:,8],axis=1),axis=0)
                image_temp  = CvBridge().cv2_to_imgmsg(flipped_output_image)
                image_temp.header = header
                self.slice_pub.publish(image_temp)
                rospy.loginfo("Published Encoded-Decoded image slice")

            if self.pub_voxels == True:
                markerarray = MarkerArray()
                filtered_points = points[points[:,3]<0.3]
                marker_ctr = 0
                for markers in filtered_points:
                    marker = Marker()
                    marker.header = header
                    marker.type = marker.CUBE
                    marker.action = marker.ADD
                    marker.scale.x = 0.14
                    marker.scale.y = 0.14
                    marker.scale.z = 0.14
                    marker.color.a = 1- markers[3]
                    marker.color.r = 255
                    marker.color.g = 0
                    marker.color.b = 0
                    marker.id = marker_ctr
                    marker_ctr +=1
                    marker.pose.position.x = markers[0]
                    marker.pose.position.y = markers[1]
                    marker.pose.position.z = markers[2]
                    marker.pose.orientation.x = 1
                    marker.pose.orientation.y = 0
                    marker.pose.orientation.z = 0
                    marker.pose.orientation.w = 0
                    markerarray.markers.append(marker)
                self.voxel_pub.publish(markerarray)
                rospy.loginfo("Published Encoded-Decoded voxels")

    def find_binary_crossentropy(self, tsdf_true, tsdf_pred):
        print("Tsdf true: ", tsdf_true.shape, "\nTsdf pred: ", tsdf_pred.shape)
        #center_of_tsdf = [26,26,8]
        tsdf_true_minimal_cube = tsdf_true[23:29,23:29,6:10] #Taking the 7x7x5 cube of the pointcloud

        tsdf_pred_minimal_cube = tsdf_pred[23:29,23:29,6:10]
        
        binary_loss_matrix = tf.keras.losses.binary_crossentropy(tsdf_true_minimal_cube, tsdf_pred_minimal_cube)
        binary_loss_tensor = tf.math.reduce_mean(binary_loss_matrix)
        binary_loss = float(binary_loss_tensor)
        return binary_loss


    def find_error_in_representations(self, tsdf_true, tsdf_pred):
        print("Tsdf true: ", tsdf_true.shape, "\nTsdf pred: ", tsdf_pred.shape)
        #center_of_tsdf = [26,26,8]
        tsdf_true_minimal_cube = tsdf_true[23:29,23:29,6:10] #Taking the 7x7x5 cube of the pointcloud

        tsdf_pred_minimal_cube = tsdf_pred[23:29,23:29,6:10]
        
        binary_loss_matrix = tf.keras.losses.binary_crossentropy(tsdf_true_minimal_cube, tsdf_pred_minimal_cube)
        binary_loss_tensor = tf.math.reduce_mean(binary_loss_matrix)
        binary_loss = float(binary_loss_tensor)
        return binary_loss


    def find_closest_distance(self,tsdf_true):

        tsdf_true_minimal_cube = tsdf_true[np.where((tsdf_true[:,0] >-0.6) & (tsdf_true[:,0] < 0.6)
                    & (tsdf_true[:,1] >-0.6) & (tsdf_true[:,1] < 0.6)
                    & (tsdf_true[:,2] >-0.6) & (tsdf_true[:,2] < 0.6)
                    & (tsdf_true[:,3] >-0.02) & (tsdf_true[:,3] < 0.15))]  #0.15 from the front and a small margin of error from the back

        shortest_distance = 10
        for row in tsdf_true_minimal_cube:
            dist = np.sqrt(row[0]**2 + row[1]**2 + row[2]**2) #Euclidean distance
            if (dist < shortest_distance):
                shortest_distance = dist

        if(shortest_distance == 10):
            return 0

        return shortest_distance



def main():
    rospy.init_node('PC2_encoder')
    pc_saver = unordered_pointcloud_to_latent_space()
    rospy.spin()



if __name__ == '__main__':
    main()