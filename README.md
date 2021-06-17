# 3D Pointcloud Convolutional Variational Autoencoder


![plot](/illustrations/convolutional_variational_autoencoder.png)

This repository contains the framework needed for collecting TSDF pointclouds, shuffling them, and using them for training a 3D Convolutional Variational Autoencoder (CVAE). 
We extract the Encoder-part of this CVAE structure to be used with a Learning-Based agent which is trained to navigate and control a MAV for executing collision-free paths. This framework also integrates functionality for monitoring the performance of the CVAE in real-time by visualizing the reconstructed pointcloud from the Decoder-part of the network. 

The pipeline works as follows:
1. First, the pointclouds are collected using the **PC2_to_3D_array_sub.py**-script. This script will collect the TSDF pointclouds coming from the TSDF_server of the VoxBlox software package. It is very important for the training process of the CVAE that these pointclouds are collected in a well-distributed manner. If the dataset of pointclouds is biased because of the MAV visiting a certain region of an environment more often, or the MAV not exploring the environment properly and only navigating in optimal paths, this bias will be passed on to the CVAE during training. Therefore, make sure that the paths the MAV takes, as well as the placement of obstacles, are as random as possible. It is also important that the MAV also explores non-optimal paths. 
The validity of the validation dataset used for monitoring overfitting also needs to be ensured in this step. Make sure that this validation dataset is completely free of examples which are similar to the ones in the training set to avoid data-leakage. 

2. The second step consists in shuffling the dataset across training batches using **dataset_shuffler.py**. This makes sure that all the different pickle-files are randomized across the whole dataset. This means that the training of the CVAE itself will be more stable, and that one can vary the number of training batches fed to the CVAE because they all have the same distribution of pointcloud samples. 

3. The third step consists in training the CVAE using the script **pointcloud_vae_trainer.py**. The CVAE itself is defined in the script **Convolutional_variational_autoencoder.py**, and this is where one can make changes of the CVAE structure before proceeding to the training process. The training process can be visualized using Tensorboard, and here one can observe the training reconstruction loss, training KL-loss, training total loss, validation reconstruction loss, validation KL-loss, validation total loss. Training should not proceed after the training loss and the validation loss have diverged.

4. The last step is loading the trained weights into the CVAE model in the **TSDF_encoder.py**. This ROS-node will perform the real-time encoding of TSDF pointclouds from the TSDF_server of the Voxblox package. The compressed latent space representation is published on a topic that a Learning-Based agent subscribes to. This node also includes functionality for real-time visualization of the reconstructed pointcloud using the Decoder of the CVAE structure. This serves as a strong indication as to what features are present in the compressed latent space representation. 


Input pointcloud from the TSDF_server       |  Reconstructed pointcloud from the CVAE
:-------------------------:|:-------------------------:
![plot](/illustrations/IL_perfect_tsdf.png)  |  ![plot](/illustrations/IL_perfect_recon.png)


### Generation of training environment and shuffling of objects
A secondary functionality included in this repository is the generation of obstacles to populate an empty .world-file. This includes a script that writes the XML-code for obstacles of different shapes and sizes, which then can be added to the .world-file used for simulation.
A script for randomly scrambling the positions of the obstacles is also included. This takes the form of a ROS-service that can be called during the collection of pointclouds, or from the Learning-Based training process, so that the environment may be scrambled at certain intervals. 

## Versions
ROS1 Melodic Morenia

Python 3.6.9

Tensorflow 2.4.1