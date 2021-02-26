
##Setup
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import matplotlib.pyplot as plt
print()
print("Setting logger level")
print()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

##Create a sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

##Build the encoder
latent_dim = 10

encoder_inputs = keras.Input(shape=(64, 64, 20, 1))
x = layers.Conv3D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv3D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

##Build the decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16*16*5*64, activation="relu")(latent_inputs)
x = layers.Reshape((16, 16, 5, 64))(x)
x = layers.Conv3DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv3DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv3DTranspose(1, 3, activation="relu", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

##Define the VAE as a model with a custom train_step
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

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
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
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

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
print("Iris dataset:")
print(x_train.shape)
print(type(x_train))

# mnist_digits = np.concatenate([x_train, x_test], axis=0)
# mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

with open('../pickelled/pointclouds.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    pointcloud_array = np.array(pickle.load(f))

# training_set = pointcloud_array[:70]
# test_set = pointcloud_array[70:]

print("Pointcloud dataset:")
print(pointcloud_array.shape)
print(type(pointcloud_array))

pointcloud_array = np.reshape(pointcloud_array,(100,65,65,20,1))

print("Pointcloud reshaped dataset:")
print(pointcloud_array.shape)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(pointcloud_array[:,:64,:64,:], epochs=1, batch_size=64)
print(pointcloud_array[:,:64,:64,:,:].shape)
print(pointcloud_array[:,:64,:64,:].shape)
print(pointcloud_array[0,:64,:64,:,:].shape)
input_pc = np.array([pointcloud_array[0,:64,:64,:,:]])
print(input_pc.shape)

latent_space= vae.encoder.predict(input_pc)
print(f'Latent space z_mean: {latent_space[0]}, \n z_log_var:{latent_space[1]} \n and z: {latent_space[2]}')
test_image = vae.decoder.predict(latent_space[2])[0,:,:,:]
print(test_image.shape)
for im in range(test_image.shape[2]):
    plt.imshow(test_image[:,:,im], cmap="gray") 
    plt.show()

