##Setup
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Convolutional_variational_autoencoder as CVAE

##Load dataset
with open('../pickelled/pointclouds.pickle', 'rb') as f:
    pointcloud_array = np.array(pickle.load(f))

pointcloud_array = np.reshape(pointcloud_array,(100,65,65,20,1))

#Shuffle dataset
np.random.shuffle(pointcloud_array)

##Construct autoencoder and train
vae = CVAE.VAE()
vae.compile(optimizer=keras.optimizers.Adam())
vae.save_weights(vae.checkpoint_path.format(epoch=0))
vae.fit(pointcloud_array[:,:64,:64,:], epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback])

##Do inference to monitor results
input_pc = np.array([pointcloud_array[0,:64,:64,:,:]])
latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]

for im in range(test_image.shape[2]):
    plt.imshow(test_image[:,:,im], cmap="gray") 
    plt.show()

