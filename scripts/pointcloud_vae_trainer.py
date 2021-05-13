##Setup
import pickle5 as pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Convolutional_variational_autoencoder as CVAE
import Minimal_convolutional_variational_autoencoder as MCVAE
import argparse
import math
import glob
import os
import time


#Load the batches of training data

train_path = path = "../pickelled/randomized_11_may/training/shuffled"
train_data = []
num_train_batches = 10

for i in range(num_train_batches):
    with open(train_path+f"/pointclouds_batch{i}.pickle",'rb') as f:
        arr = np.array(pickle.load(f))
        train_data.append(arr)
        print(f'Imported training pickle {i}')

training_dataset = np.reshape(train_data, (1000*num_train_batches,65,65,16,1))

#Load the validation dataset

val_path = "../pickelled/randomized_11_may/validation"
validation_data = []
num_val_batches = 2
for i in range(num_val_batches):
    with open(val_path+f"/shuffled/pointclouds_batch{i}.pickle",'rb') as f:
        arr = np.array(pickle.load(f))
        validation_data.append(arr)
        print(f'Imported validation pickle {i}')

validation_dataset = np.reshape(validation_data, (1000*num_val_batches,65,65,16,1))

#Construct autoencoder
#print(f'len pkl files:{len(pkl_files[0])}')
vae = CVAE.VAE(dataset_size= len(training_dataset)) #Set the correct batch count for saving frequency
vae.compile(optimizer=vae.optimizer)
print(vae.dataset_size)
print(vae.batch_count)

np.random.seed(int(time.time()))
random_index = [int(np.random.uniform(low=0,high=1000*num_train_batches)) for x in range(15)]
print(random_index)

#Inspection of input data
for i in random_index:
    plt.imshow(training_dataset[i,:64,:64,8,0], cmap="gray") 
    plt.show()


#Train
vae.save_weights(vae.checkpoint_path.format(epoch=0))
vae.write_summary_to_file()
vae.fit(training_dataset[:,:64,:64,:,:],validation_data=(validation_dataset[:,:64,:64,:,:],validation_dataset[:,:64,:64,:,:]),sample_weight=np.array([0.6,0.4]), epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])

##Do inference to monitor results
input_pc = np.array([validation_dataset[0,:64,:64,:,:]])
latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]

for im in range(test_image.shape[2]):
    plt.imshow(test_image[:,:,im], cmap="gray") 
    plt.show()

