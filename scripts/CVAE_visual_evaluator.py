##Setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Convolutional_variational_autoencoder as CVAE
import argparse

pointcloud_list = []
#num_batches = 1 #How many batches of the dataset to load
#test_im_index = 719 #Image to be evaluated
test_images_indices = [0,8,14,19]
#if test_im_index > num_batches*1000:
#   print("Test_im_index higher than the number of pointclouds from the loaded batches")

##Load dataset
#for i in range(num_batches):
with open('../pickelled/test_set_4envs/test_set_4envs.pickle', 'rb') as f:
    pointcloud_list.append(np.array(pickle.load(f)))

pointcloud_array = np.reshape(pointcloud_list,(20,65,65,20,1))

#Need to reshape array for compatibility with triple conv encoder
new_pc_array = np.zeros((20,65,65,24,1))
new_pc_array[:,:,:,:20,:] = pointcloud_array
pointcloud_array = new_pc_array

#Clear from memory
del new_pc_array

print(pointcloud_array.shape)

##Construct autoencoder
vae = CVAE.VAE()

#Load weights?
parser = argparse.ArgumentParser(description="Model weight parser")
parser.add_argument('model_weight_dir',type=str,help='The path for the autoencoder weights')
args = parser.parse_args()
arg_filepath = args.model_weight_dir
vae.load_weights(arg_filepath)

output_images = []

##Do inference to monitor results
for i in range(20):
    input_pc = np.array([pointcloud_array[i,:64,:64,:,:]])
    latent_space= vae.encoder.predict(input_pc)
    output_images.append(vae.decoder.predict(latent_space[2])[0,:,:,:])

output_images = np.array(output_images)

for indx in test_images_indices:
    #for im in range(20):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(pointcloud_array[indx,:64,:64,8,0], cmap="gray")
    axarr[0].set_title('Input image')
    axarr[1].imshow(output_images[indx,:,:,8,0], cmap="gray")
    axarr[1].set_title('Encoded-Decoded image')
    #f.suptitle(f'In', fontsize=12)
    f.savefig(f'../eval_images/Latent_dim_{vae.latent_dim}_test_ind_{indx}_8.eps')
        


