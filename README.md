**Pointcloud Convolutional Variational Autoencoder**

This node receives Pointcloud2 messages from ROS with Local Truncated Signed Distance Field (TSDF) representations. The
autoencoder trains on this pointcloud data and minimizes the reprojection error. The goal is to have a node that receives pointclouds
and outputs the latent space to be used in Reinforcement learning agent later. 
