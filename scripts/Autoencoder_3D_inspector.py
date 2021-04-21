#!/usr/bin/env python

#rosrun pointcloud_utils Autoencoder_3D_inspector.py src/pointcloud_utils/saved_model_weights/latent_dim_30/24-03_10:44/epoch_0031/cp-.ckpt

import rospy
import numpy as np
import ros_numpy as ros_np
import argparse
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Convolutional_variational_autoencoder as CVAE

class autoencoder_pc_reconstruction():
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
        xyzi = np.zeros((65,65,16))

        #Convert from pointcloud2 to numpy array
        arr = np.array(ros_np.point_cloud2.pointcloud2_to_array(pc).tolist())

        #Convert from decimal position values, to enumerated index values
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


        #Find the scale factor
        scale_factor = np.abs(x_unique[0]-x_unique[1])

        #Fill numpy array with the correct intensity value at each index
        xyzi[x_enum,y_enum,z_enum]= arr[:,3]

        #Threshold the input image
        binarized = np.where(xyzi > self.bin_thresh, 1,0)

        #Format for TF
        tf_input = np.reshape(binarized[:64,:64,:],(64,64,16,1))

        #Do inference
        input_pc = np.array([tf_input])
        latent_space = self.vae.encoder.predict(input_pc)
        output_image = self.vae.decoder.predict(latent_space[0])[0,:,:,:]

        #Convert output image to list of points
        output_image = np.reshape(output_image,(64,64,24))
        cropped_output_image = output_image[:54,:54,:16]
        m,n,l = cropped_output_image.shape
        R,C,T = np.mgrid[:m,:n,:l]*scale_factor

        #Shift points to that the origin matches
        R = R - (scale_factor*27)
        C = C - (scale_factor*27)
        T = T - (scale_factor*10)

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

def main():
    rospy.init_node('PC2_encoder')
    pc_saver = autoencoder_pc_reconstruction()
    rospy.spin()

if __name__ == '__main__':
    main()