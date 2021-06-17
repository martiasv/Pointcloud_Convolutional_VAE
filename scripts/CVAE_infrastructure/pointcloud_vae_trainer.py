##Setup
import pickle5 as pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Convolutional_variational_autoencoder as CVAE
import argparse
import math
import glob
import os
import time

"""This script runs the whole training process. After having pickled the pointclouds using PC2_to_3Darray_sub.py, and scrambled them using dataset_shuffled.py,
they can be loaded into this script to be used for training."""

#Load the batches of training data
train_path = path = "../pickelled/randomized_11_may/training/shuffled"
train_data = []
num_train_batches = 2

#Unpickle the files and append the data to the training 
for i in range(num_train_batches):
    with open(train_path+f"/pointclouds_batch{i}.pickle",'rb') as f:
        arr = np.array(pickle.load(f))
        train_data.append(arr)
        print(f'Imported training pickle {i}')

#Reshape for the correct dimensions for training using TF
training_dataset = np.reshape(train_data, (1000*num_train_batches,65,65,16,1))

#Load the validation dataset
val_path = "../pickelled/randomized_11_may/validation"
validation_data = []
num_val_batches = 1
for i in range(num_val_batches):
    with open(val_path+f"/shuffled/pointclouds_batch{i}.pickle",'rb') as f:
        arr = np.array(pickle.load(f))
        validation_data.append(arr)
        print(f'Imported validation pickle {i}')

#Reshape for the correct dimensions for validation using TF
validation_dataset = np.reshape(validation_data, (1000*num_val_batches,65,65,16,1))

#Construct autoencoder
vae = CVAE.VAE(dataset_size= len(training_dataset)) #Set the correct batch count for saving frequency
vae.compile(optimizer=vae.optimizer)
print(f"The size of the dataset is: {vae.dataset_size}")
print(f"The number of batches is:{vae.batch_count}")

#Generate random numbers for sanity-checking the dataset
np.random.seed(int(time.time()))
random_index = [int(np.random.uniform(low=0,high=1000*num_train_batches)) for x in range(15)]
print(random_index)

#Inspection of input data. If there has been any error in the pipeline before training, this should be discovered in this step
for i in random_index:
    plt.imshow(training_dataset[i,:64,:64,8,0], cmap="gray") 
    plt.show()


#Train the CVAE using both training and validation dataset. Saving the model weights happen at an interval defined in the CVAE script
vae.save_weights(vae.checkpoint_path.format(epoch=0))
vae.write_summary_to_file()
vae.fit(training_dataset[:,:64,:64,:,:],validation_data=(validation_dataset[:,:64,:64,:,:],validation_dataset[:,:64,:64,:,:]),sample_weight=np.array([0.6,0.4]), epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])

##Do inference to monitor results. This is to see if there has been any error in the training process, and that the CVAE functions properly
input_pc = np.array([validation_dataset[0,:64,:64,:,:]])
latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]

for im in range(test_image.shape[2]):
    plt.imshow(test_image[:,:,im], cmap="gray") 
    plt.show()

