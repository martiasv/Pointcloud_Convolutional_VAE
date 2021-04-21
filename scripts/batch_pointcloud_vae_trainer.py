##Setup
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Convolutional_variational_autoencoder as CVAE
import Minimal_convolutional_variational_autoencoder as MCVAE
import argparse
import math


pointcloud_list = []
num_batches = 1
envs = ['test_corridor','randomized',"randomized_2"]#,'randomized']#,'tunnel','large_obstacles']

## BATCH 1
print()
print("BATCH 1")
print()

##Load dataset
for env in envs:
    with open("../pickelled/"+env+"/shuffled/pointclouds_batch"+str(1)+'.pickle', 'rb') as f:
        array = np.array(pickle.load(f))
        print(f'Env {env} batch 1 shape: {array.shape}')
        pointcloud_list.append(array)

pointcloud_array = np.reshape(pointcloud_list,(1000*len(envs),65,65,16,1))


#Need to reshape array for compatibility with triple conv encoder
new_pc_array = np.zeros((1000*len(envs),65,65,32,1))
new_pc_array[:,:,:,:20,:] = pointcloud_array
pointcloud_array = new_pc_array

#Clear from memory
del new_pc_array

print(pointcloud_array.shape)

#Construct autoencoder
vae = MCVAE.VAE(dataset_size= pointcloud_array.shape[0]) #Set the correct batch count for saving frequency
vae.compile(optimizer=vae.optimizer)

print(vae.batch_count)

#Shuffle dataset
np.random.shuffle(pointcloud_array)

random_index = [np.random.uniform(low=0,high=1000*num_batches) for x in range(15)]

#Inspection of input data
for i in range(len(random_index)):
    plt.imshow(pointcloud_array[i,:64,:64,8,0], cmap="gray") 
    plt.show()


#Train
vae.save_weights(vae.checkpoint_path.format(epoch=0))
vae.write_summary_to_file()
vae.fit(pointcloud_array[:,:64,:64,:],validation_split=vae.validation_split, epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])

#Clear RAM
del pointcloud_list
del pointcloud_array
pointcloud_list = []

##BATCH 2####
print()
print("BATCH 2")
print()

##Load dataset
for env in envs:
    with open("../pickelled/"+env+"/shuffled/pointclouds_batch"+str(2)+'.pickle', 'rb') as f:
        array = np.array(pickle.load(f))
        print(f'Env {env} batch 2 shape: {array.shape}')
        pointcloud_list.append(array)

pointcloud_array = np.reshape(pointcloud_list,(1000*len(envs),65,65,16,1))


#Need to reshape array for compatibility with triple conv encoder
new_pc_array = np.zeros((1000*len(envs),65,65,32,1))
new_pc_array[:,:,:,:20,:] = pointcloud_array
pointcloud_array = new_pc_array

#Clear from memory
del new_pc_array

print(pointcloud_array.shape)

#Shuffle dataset
np.random.shuffle(pointcloud_array)

#Train
vae.save_weights(vae.checkpoint_path.format(epoch=vae.epochs+1))
vae.write_summary_to_file()
vae.fit(pointcloud_array[:,:64,:64,:],validation_split=vae.validation_split, epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])

#Clear RAM
del pointcloud_list
del pointcloud_array
pointcloud_list = []

##BATCH 3####
print()
print("BATCH 3")
print()

##Load dataset
for env in envs:
    with open("../pickelled/"+env+"/shuffled/pointclouds_batch"+str(3)+'.pickle', 'rb') as f:
        array = np.array(pickle.load(f))
        print(f'Env {env} batch 3 shape: {array.shape}')
        pointcloud_list.append(array)

pointcloud_array = np.reshape(pointcloud_list,(1000*len(envs),65,65,16,1))


#Need to reshape array for compatibility with triple conv encoder
new_pc_array = np.zeros((1000*len(envs),65,65,32,1))
new_pc_array[:,:,:,:20,:] = pointcloud_array
pointcloud_array = new_pc_array

#Clear from memory
del new_pc_array

print(pointcloud_array.shape)

#Shuffle dataset
np.random.shuffle(pointcloud_array)

#Train
vae.save_weights(vae.checkpoint_path.format(epoch=2*vae.epochs+1))
vae.write_summary_to_file()
vae.fit(pointcloud_array[:,:64,:64,:],validation_split=vae.validation_split, epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])

#Clear RAM
del pointcloud_list
del pointcloud_array
pointcloud_list = []

##BATCH 4####
print()
print("BATCH 4")
print()

##Load dataset
for env in envs:
    with open("../pickelled/"+env+"/shuffled/pointclouds_batch"+str(4)+'.pickle', 'rb') as f:
        array = np.array(pickle.load(f))
        print(f'Env {env} batch 4 shape: {array.shape}')
        pointcloud_list.append(array)

pointcloud_array = np.reshape(pointcloud_list,(1000*len(envs),65,65,16,1))


#Need to reshape array for compatibility with triple conv encoder
new_pc_array = np.zeros((1000*len(envs),65,65,32,1))
new_pc_array[:,:,:,:20,:] = pointcloud_array
pointcloud_array = new_pc_array

#Clear from memory
del new_pc_array

print(pointcloud_array.shape)

#Shuffle dataset
np.random.shuffle(pointcloud_array)

#Train
vae.save_weights(vae.checkpoint_path.format(epoch=3*vae.epochs+1))
vae.write_summary_to_file()
vae.fit(pointcloud_array[:,:64,:64,:],validation_split=vae.validation_split, epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])




##Do inference to monitor results
input_pc = np.array([pointcloud_array[0,:64,:64,:,:]])
latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]

for im in range(test_image.shape[2]):
    plt.imshow(test_image[:,:,im], cmap="gray") 
    plt.show()

