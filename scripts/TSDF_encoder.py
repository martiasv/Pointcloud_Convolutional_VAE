#!/usr/bin/env python

#rosrun pointcloud_utils TSDF_encoder.py "./src/pointcloud_utils/saved_model_weights/26-2_13:1/0015/cp-.ckpt"


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
import Convolutional_variational_autoencoder as CVAE
from std_msgs.msg import Float32MultiArray
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pointcloud_utils.msg import LatSpace

class unordered_pointcloud_to_latent_space():
    def __init__(self,pc_topic="/gagarin/tsdf_server/tsdf_pointcloud",lat_topic="/gagarin/pc_latent_space"):

        #Parse arguments
        self.parser = argparse.ArgumentParser(description="Model weight parser")
        self.parser.add_argument('model_weight_dir',type=str,help='The path for the autoencoder weights')
        self.args = self.parser.parse_args()
        self.arg_filepath = self.args.model_weight_dir

        #Load the network
        self.vae = CVAE.VAE()
        self.vae.compile(optimizer=keras.optimizers.Adam())
        self.vae.load_weights(self.arg_filepath)
        self.latent_space_dim = self.vae.latent_dim

        #Make subscriber and publisher
        self.pc_sub = rospy.Subscriber(pc_topic,PointCloud2,self.point_cloud_encoder_callback)
        self.pc_pub = rospy.Publisher(lat_topic,LatSpace,queue_size=None)

    def point_cloud_encoder_callback(self,pc):
        #Initalize empty array for TSDF to fill
        xyzi = np.zeros((65,65,24))

        #Convert from pointcloud2 to numpy array
        arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())

        #Convert from decimal position values, to enumerated index values
        _,x_enum = np.unique(arr[:,0],return_inverse= True)
        _,y_enum = np.unique(arr[:,1],return_inverse= True)
        _,z_enum = np.unique(arr[:,2],return_inverse= True)

        #Fill numpy array with the correct intensity value at each index
        xyzi[x_enum,y_enum,z_enum]= arr[:,3]

        #Format for TF
        tf_input = np.reshape(xyzi[:64,:64,:],(64,64,24,1))

        #Do inference
        input_pc = np.array([tf_input])
        latent_space = self.vae.encoder.predict(input_pc)[0].tolist()[0]

        #Publish the result
        msg = LatSpace(latent_space=latent_space)
        self.pc_pub.publish(msg)



def main():
    rospy.init_node('PC2_encoder')
    pc_saver = unordered_pointcloud_to_latent_space()
    rospy.spin()



if __name__ == '__main__':
    main()