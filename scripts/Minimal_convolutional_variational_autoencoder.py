import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import date, time, datetime
import os
import io
import math
from scipy import signal, ndimage
import matplotlib.pyplot as plt

##Create a sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


##Define the VAE as a model with a custom train_step
class VAE(keras.Model):
    def __init__(self, dataset_size=20, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = 30
        self.blurring_kernel = np.ones((3,3,3,1,1))
        #self.tensor_input_shape = (64, 64, 24, 1)
        self.batch_size = 64
        self.epochs  = 32
        self.activation_function = "relu"
        self.output_activation_function = "sigmoid"
        self.kernel_size = 3
        self.strides = 2
        self.padding = "same"
        self.encoder_conv_filters = [8,16]
        self.encoder_dense_layers = [128,64]
        self.decoder_conv_filters = [16,8]
        self.decoder_dense_layers = [64,128,8*8*4*16]
        self.save_freq = 8 #Save after this many epochs
        self.dataset_size = dataset_size
        self.validation_split = 0.2
        self.batch_count = math.ceil((self.dataset_size*(1-self.validation_split))/self.batch_size)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        #Define training loss functions
        self.loss_function = keras.losses.BinaryCrossentropy
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.learning_rate = 0.001
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.base_path = "../saved_model_weights/latent_dim_"+str(self.latent_dim)+"/"+ f'{datetime.now().day:02d}-{datetime.now().month:02d}_{datetime.now().hour:02d}:{datetime.now().minute:02d}'
        self.checkpoint_path = self.base_path + "/epoch_{epoch:04d}/cp-.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.dataset_dir = '../pickelled/tunnel/shuffled/pointclouds_batch'
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,save_freq=self.save_freq*self.batch_count)
        self.logdir = "../logs/latent_dim_"+str(self.latent_dim)+"/"+ f'{datetime.now().day:02d}-{datetime.now().month:02d}_{datetime.now().hour:02d}:{datetime.now().minute:02d}'
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)


    def set_batch_count(self,count):
        self.batch_count = count 
    
    def build_encoder(self):
        encoder_inputs = keras.Input(shape=(64, 64, 32, 1))
        x = encoder_inputs
        #for idx in range(len(self.encoder_conv_filters)):
        x = layers.Conv3D(self.encoder_conv_filters[0], self.kernel_size, activation=self.activation_function, strides=self.strides, padding=self.padding)(x)
        #x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
        x = layers.Conv3D(self.encoder_conv_filters[1], self.kernel_size, activation=self.activation_function, strides=self.strides, padding=self.padding)(x)
        x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.encoder_dense_layers[0])(x)
        x = layers.Dense(self.encoder_dense_layers[1])(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.decoder_dense_layers[0], activation=self.activation_function)(latent_inputs)
        x = layers.Dense(self.decoder_dense_layers[1], activation=self.activation_function)(x)
        x = layers.Dense(self.decoder_dense_layers[2], activation=self.activation_function)(x)
        x = layers.Reshape((8, 8, 4, 16))(x)
        x = layers.UpSampling3D(size=(2,2,2))(x)
        x = layers.Conv3DTranspose(self.decoder_conv_filters[0], self.kernel_size, activation=self.activation_function, strides=self.strides, padding=self.padding)(x)
        x = layers.Conv3DTranspose(self.decoder_conv_filters[1], self.kernel_size, activation=self.activation_function, strides=self.strides, padding=self.padding)(x)
        decoder_outputs = layers.Conv3DTranspose(1, self.kernel_size, activation=self.output_activation_function, padding=self.padding)(x) #Stride 1 for collapsing into correct dimensions
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    self.loss_function(data, reconstruction), axis=(1, 2)
                )
            ) #+ add term for gaussian filtered edge detection, 
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    def call(self,data):
        z_mean, z_log_var, z = self.encoder(data)
        return self.decoder(z_mean)

    def test_step(self,data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                self.loss_function(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return {"loss":total_loss,"reconstruction_loss":reconstruction_loss,"kl_loss":kl_loss}

    def write_summary_to_file(self):
        file1 = open(self.base_path+"/network_and_training_summary.txt","w")
        Data_to_save = f'Latent space dimension:{self.latent_dim}\nActivation function:'+self.activation_function+f'\nConvolution kernel size:{self.kernel_size}\nConvolution strides:{self.strides}\nPadding:{self.padding}\nBatch size:{self.batch_size}\nEpochs:{self.epochs}\nOptimizer:{type(self.optimizer)}\nLearning rate:{self.learning_rate}\nLoss function: {self.loss_function}\nDataset dir:{self.dataset_dir}\n\nAutoencoder network summary:\nEncoder convolutional filters:{self.encoder_conv_filters}\nEncoder dense layers:{self.encoder_dense_layers}\n\nDecoder Convolutional filters:{self.decoder_conv_filters}\nDecoder dense layers:{self.decoder_dense_layers}\n'
        stringlist = []
        self.encoder.summary(line_length=120,print_fn=lambda x: stringlist.append(x))
        self.decoder.summary(line_length=120,print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        file1.writelines(Data_to_save+model_summary)
        file1.close()

    def sobel_edge_3d(self,inputTensor):
        # This function computes Sobel edge maps on 3D images
        # inputTensor: input 3D images, with size of [batchsize,W,H,D,1]
        # output: output 3D edge maps, with size of [batchsize,W-2,H-2,D-2,3], each channel represents edge map in one dimension
        sobel1 = tf.constant([0.1,0,-0.1],tf.float32) # 1D edge filter
        sobel2 = tf.constant([0.1,0.2,0.1],tf.float32) # 1D blur weight
        
        # generate sobel1 and sobel2 on x- y- and z-axis, saved in sobel1xyz and sobel2xyz
        sobel1xyz = [sobel1,sobel1,sobel1]
        sobel2xyz = [sobel2,sobel2,sobel2]
        for xyz in range(3):
            newShape = [1,1,1,1,1]
            newShape[xyz] = 3
            sobel1xyz[xyz] = tf.reshape(sobel1,newShape)
            sobel2xyz[xyz] = tf.reshape(sobel2,newShape)
            
        # outputTensor_x will be the Sobel edge map in x-axis
        outputTensor_x = tf.nn.conv3d(inputTensor,sobel1xyz[0],strides=[1,1,1,1,1],padding='SAME') # edge filter in x-axis
        outputTensor_x = tf.nn.conv3d(outputTensor_x,sobel2xyz[1],strides=[1,1,1,1,1],padding='SAME') # blur filter in y-axis
        outputTensor_x = tf.nn.conv3d(outputTensor_x,sobel2xyz[2],strides=[1,1,1,1,1],padding='SAME') # blur filter in z-axis
        
        outputTensor_y = tf.nn.conv3d(inputTensor,sobel1xyz[1],strides=[1,1,1,1,1],padding='SAME')
        outputTensor_y = tf.nn.conv3d(outputTensor_y,sobel2xyz[0],strides=[1,1,1,1,1],padding='SAME')
        outputTensor_y = tf.nn.conv3d(outputTensor_y,sobel2xyz[2],strides=[1,1,1,1,1],padding='SAME')
        
        outputTensor_z = tf.nn.conv3d(inputTensor,sobel1xyz[2],strides=[1,1,1,1,1],padding='SAME')
        outputTensor_z = tf.nn.conv3d(outputTensor_z,sobel2xyz[0],strides=[1,1,1,1,1],padding='SAME')
        outputTensor_z = tf.nn.conv3d(outputTensor_z,sobel2xyz[1],strides=[1,1,1,1,1],padding='SAME')
        
        return tf.concat([outputTensor_x,outputTensor_y,outputTensor_z],4)