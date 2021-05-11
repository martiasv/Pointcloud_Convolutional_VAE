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
import glob
import os

def load_batches(name):
    decoded = name.decode("UTF-8")
    if os.path.exists(decoded):
        with open(decoded,'rb') as f:
            empty_pcl = np.zeros((65,65,16,1))
            loaded_pickle = pickle.load(f)
            empty_pcl = loaded_pickle
            
            return loaded_pickle[:,:64,64,:,:]

#Load the batches of training data

path = "../pickelled/test_corridor_yawless_spawn"
pkl_files = glob.glob(path+"/poinclouds_batch*.pickle")
dataset = tf.data.Dataset.from_tensor_slices((pkl_files))
dataset = dataset.map(
    lambda filename: tuple(tf.py_function(load_batches, [filename], tf.float32))
)

#Load the validation dataset

val_path = "../pickelled/test_corridor_yawless_spawn_validation/"
validation_data = []
num_val_batches = 10
for i in range(num_val_batches):
    with open(val_path+f"/shuffled/pointclouds_batch{i}.pickle") as f:
        validation_data.append(np.array(pickle.load(f)))

validation_dataset = np.reshape(validation_data, (1000*num_val_batches,65,65,16,1))

#Construct autoencoder
vae = MCVAE.VAE(dataset_size= 1000*len(pkl_files)) #Set the correct batch count for saving frequency
vae.compile(optimizer=vae.optimizer)

print(vae.batch_count)

random_index = [np.random.uniform(low=0,high=1000*num_val_batches) for x in range(15)]

#Inspection of input data
for i in range(len(random_index)):
    plt.imshow(validation_dataset[i,:64,:64,8,0], cmap="gray") 
    plt.show()


#Train
vae.save_weights(vae.checkpoint_path.format(epoch=0))
vae.write_summary_to_file()
vae.fit(dataset,validation_data=validation_dataset[:,:64,:64,:], epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])

##Do inference to monitor results
input_pc = np.array([validation_dataset[0,:64,:64,:,:]])
latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]

for im in range(test_image.shape[2]):
    plt.imshow(test_image[:,:,im], cmap="gray") 
    plt.show()

