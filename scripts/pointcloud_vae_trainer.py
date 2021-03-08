##Setup
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Convolutional_variational_autoencoder as CVAE
import argparse

pointcloud_list = []
num_batches = 3

##Load dataset
for i in range(num_batches):
    with open('../pickelled/decorated_virginia_mine/pointclouds_batch'+str(i)+'.pickle', 'rb') as f:
        pointcloud_list.append(np.array(pickle.load(f)))

pointcloud_array = np.reshape(pointcloud_list,(1000*num_batches,65,65,20,1))

print(pointcloud_array.shape)

#Shuffle dataset
np.random.shuffle(pointcloud_array)

random_index = [np.random.uniform(low=0,high=1000*num_batches) for x in range(3)]

#Inspection of input data
#for im in range(pointcloud_array.shape[2]-1):
for i in range(len(random_index)):
    plt.imshow(pointcloud_array[i,:64,:64,10,0], cmap="gray") 
    plt.show()

#Construct autoencoder
vae = CVAE.VAE()
vae.compile(optimizer=vae.optimizer)

#Load weights?
# parser = argparse.ArgumentParser(description="Model weight parser")
# parser.add_argument('model_weight_dir',type=str,help='The path for the autoencoder weights')
# args = parser.parse_args()
# arg_filepath = args.model_weight_dir
# if arg_filepath:    
#     vae.load_weights(arg_filepath)
#     print("Continuing training from previous session")
# else:
#     print("Training from scratch")

#Train
vae.save_weights(vae.checkpoint_path.format(epoch=0))
vae.write_summary_to_file()
vae.fit(pointcloud_array[:,:64,:64,:], epochs=vae.epochs, batch_size=vae.batch_size,callbacks=[vae.cp_callback, vae.tensorboard_callback])

##Do inference to monitor results
input_pc = np.array([pointcloud_array[0,:64,:64,:,:]])
latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]

for im in range(test_image.shape[2]):
    plt.imshow(test_image[:,:,im], cmap="gray") 
    plt.show()

