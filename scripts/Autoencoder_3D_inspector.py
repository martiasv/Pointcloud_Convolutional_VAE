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
from std_msgs.msg import Header
from sensor_msgs.msg import PointField

class unordered_pointcloud_to_latent_space():
    def __init__(self,pc_topic="/gagarin/tsdf_server/tsdf_pointcloud",recon_pc="/gagarin/reconstructed_pc"):

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
        self.pc_pub = rospy.Publisher(recon_pc,PointCloud2,queue_size=1)

    def point_cloud_encoder_callback(self,pc):
        #Initalize empty array for TSDF to fill
        xyzi = np.zeros((65,65,24))
        print(type(ros_np.point_cloud2.pointcloud2_to_array(pc)))
        print(ros_np.point_cloud2.pointcloud2_to_array(pc).shape)
        print(type(ros_np.point_cloud2.pointcloud2_to_array(pc)[0]))
        print(ros_np.point_cloud2.pointcloud2_to_array(pc)[0])
        #Convert from pointcloud2 to numpy array
        arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())
        print(type(arr))
        print(arr.shape)
        #Convert from decimal position values, to enumerated index values
        x_unique,x_enum = np.unique(arr[:,0],return_inverse= True)
        _,y_enum = np.unique(arr[:,1],return_inverse= True)
        _,z_enum = np.unique(arr[:,2],return_inverse= True)

        #Find the scale factor
        scale_factor = np.abs(x_unique[0]-x_unique[1])

        #Fill numpy array with the correct intensity value at each index
        xyzi[x_enum,y_enum,z_enum]= arr[:,3]

        #Format for TF
        tf_input = np.reshape(xyzi[:64,:64,:],(64,64,24,1))

        #Do inference
        input_pc = np.array([tf_input])
        print(f'Input shape: {input_pc.shape}')
        latent_space = self.vae.encoder.predict(input_pc)
        #print(f'Latent shape: {latent_space.shape}')
        output_image = self.vae.decoder.predict(latent_space[2])[0,:,:,:]
        print(f'Output shape: {output_image.shape}')

        #Convert output image to record array
        output_image = np.reshape(output_image,(64,64,24))
        m,n,l = output_image.shape
        R,C,T = np.mgrid[:m,:n,:l]
        #output_record_array = np.column_stack((C.ravel(),R.ravel(),T.ravel(), output_image.ravel()))

        # #Scale XYZ to match the correct dimensions
        # output_record_array[:,:3] = output_record_array[:,:3]*scale_factor

        # #Shift to match centers
        # #Shift X-Y
        # output_record_array[:,:1] = output_record_array[:,:1]-(scale_factor*32)

        # #Shift Z
        # output_record_array[:,2] = output_record_array[:,2]-(scale_factor*10)
        # print(output_record_array.shape)

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
        points = np.array([C.ravel(),R.ravel(),T.ravel(),output_image.ravel()]).reshape(4,-1).T

        #Create the PointCloud2
        output_pc = pc2.create_cloud(header,fields,points)

        self.pc_pub.publish(output_pc)
        rospy.loginfo("Published Encoded-Decoded pointcloud")



def main():
    rospy.init_node('PC2_encoder')
    pc_saver = unordered_pointcloud_to_latent_space()
    rospy.spin()



if __name__ == '__main__':
    main()