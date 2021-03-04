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
num_batches = 1

##Load dataset
for i in range(num_batches):
    with open('../pickelled/decorated_virginia_mine/pointclouds_batch'+str(i)+'.pickle', 'rb') as f:
        pointcloud_list.append(np.array(pickle.load(f)))

pointcloud_array = np.reshape(pointcloud_list,(1000*num_batches,65,65,20,1))

print(pointcloud_array.shape)

##Construct autoencoder
vae = CVAE.VAE()
vae.compile(optimizer=keras.optimizers.Adam())

#Load weights?
parser = argparse.ArgumentParser(description="Model weight parser")
parser.add_argument('model_weight_dir',type=str,help='The path for the autoencoder weights')
args = parser.parse_args()
arg_filepath = args.model_weight_dir
vae.load_weights(arg_filepath)
print()
print(pointcloud_array[0,:64,:64,0,0].shape)
print()
##Do inference to monitor results
input_pc = np.array([pointcloud_array[0,:64,:64,:,:]])
latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]

for im in range(test_image.shape[2]):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(pointcloud_array[0,:64,:64,im,0], cmap="gray")
    axarr[0].set_title('Input image')
    axarr[1].imshow(test_image[:,:,im], cmap="gray")
    axarr[1].set_title('Encoded-Decoded image')
    plt.show()




# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# imgplot = plt.imshow(lum_img)
# ax.set_title('Before')
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
# ax = fig.add_subplot(1, 2, 2)
# imgplot = plt.imshow(lum_img)
# imgplot.set_clim(0.0, 0.7)
# ax.set_title('After')
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')