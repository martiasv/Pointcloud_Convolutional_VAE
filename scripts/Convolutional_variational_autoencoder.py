import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import date, time, datetime
import os
import io

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
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = 30
        #self.tensor_input_shape = (64, 64, 24, 1)
        self.batch_size = 64
        self.epochs  = 32
        self.activation_function = "relu"
        self.kernel_size = 3
        self.strides = 2
        self.padding = "same"
        self.encoder_conv_filters = [32,64]
        self.encoder_dense_layers = [64]
        self.decoder_conv_filters = [64,32]
        self.decoder_dense_layers = [8*8*3*64]
        self.save_freq = 2
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.learning_rate = 0.001
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.base_path = "../saved_model_weights/latent_dim_"+str(self.latent_dim)+"/"+ f'{datetime.now().day:02d}-{datetime.now().month:02d}_{datetime.now().hour:02d}:{datetime.now().minute:02d}'
        self.checkpoint_path = self.base_path + "/epoch_{epoch:04d}/cp-.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.dataset_dir = '../pickelled/tunnel/shuffled/pointclouds_batch'
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,save_freq=self.save_freq*self.batch_size)
        self.logdir = "../logs/latent_dim_"+str(self.latent_dim)+"/"+ f'{datetime.now().day:02d}-{datetime.now().month:02d}_{datetime.now().hour:02d}:{datetime.now().minute:02d}'
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)

    
    def build_encoder(self):
        encoder_inputs = keras.Input(shape=(64, 64, 24, 1))
        x = encoder_inputs
        for idx in range(len(self.encoder_conv_filters)):
            x = layers.Conv3D(self.encoder_conv_filters[idx], self.kernel_size, activation=self.activation_function, strides=self.strides, padding=self.padding)(x)
        x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.encoder_dense_layers[0])(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.decoder_dense_layers[0], activation=self.activation_function)(latent_inputs)
        x = layers.Reshape((8, 8, 3, 64))(x)
        x = layers.UpSampling3D(size=(2,2,2))(x)
        for idx in range(len(self.decoder_conv_filters)):
            x = layers.Conv3DTranspose(self.decoder_conv_filters[idx], self.kernel_size, activation=self.activation_function, strides=self.strides, padding=self.padding)(x)
        decoder_outputs = layers.Conv3DTranspose(1, self.kernel_size, activation=self.activation_function, padding=self.padding)(x)
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
                    keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
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


    def write_summary_to_file(self):
        file1 = open(self.base_path+"/network_and_training_summary.txt","w")
        Data_to_save = f'Latent space dimension:{self.latent_dim}\nActivation function:'+self.activation_function+f'\nConvolution kernel size:{self.kernel_size}\nConvolution strides:{self.strides}\nPadding:{self.padding}\nBatch size:{self.batch_size}\nEpochs:{self.epochs}\nOptimizer:{type(self.optimizer)}\nLearning rate:{self.learning_rate}\nDataset dir:{self.dataset_dir}\n\nAutoencoder network summary:\nEncoder convolutional filters:{self.encoder_conv_filters}\nEncoder dense layers:{self.encoder_dense_layers}\n\nDecoder Convolutional filters:{self.decoder_conv_filters}\nDecoder dense layers:{self.decoder_dense_layers}\n'
        stringlist = []
        self.encoder.summary(line_length=120,print_fn=lambda x: stringlist.append(x))
        self.decoder.summary(line_length=120,print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        file1.writelines(Data_to_save+model_summary)
        file1.close()