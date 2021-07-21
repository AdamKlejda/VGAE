from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from spektral.layers import GCNConv
import tensorflow as tf


def loss_f(y_true, y_pred,verbose):
    loss = 0
    x_true, a_true = y_true
    x_pred, a_pred = y_pred

    if verbose:
        print(x_true[0])
        print(x_pred[0])
    
        print(a_true[0])
        print(a_pred[0])
    loss += tf.reduce_mean(tf.abs(tf.cast(tf.squeeze(x_true),tf.float32) - tf.cast(tf.squeeze(x_pred),tf.float32)))
    loss += tf.reduce_mean(tf.abs(tf.cast(tf.squeeze(a_true),tf.float32) - tf.cast(tf.squeeze(a_pred),tf.float32)))
    return loss
    
class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Autoencoder(Model):
    def __init__(self, latent_dim,n_hidden,n_samples,adjency_size):
        super(Autoencoder, self).__init__()
        self.n_samples= n_samples
        self.latent_dim = latent_dim
        self.adjency_size = adjency_size
#         encoder
        self.conv1 = GCNConv(n_hidden, activation='relu')
        self.flat1 = layers.Flatten()
        self.drop1 = layers.Dropout(.2)
        self.dense1 = layers.Dense(latent_dim, activation='relu')
        self.norm = layers.BatchNormalization()

        self.z_mean = layers.Dense(latent_dim,name="z_mean")
        self.z_log_var = layers.Dense(latent_dim,name="z_log_var")
        
        # decoder A        
        self.dense2 = layers.Dense(self.adjency_size*self.adjency_size, activation='sigmoid')
        self.reshape2 = layers.Reshape((self.adjency_size, self.adjency_size))
        
        # decoder X
#         self.conv3 = GCNConv(n_hidden, activation='tanh')
        self.dense3  = layers.Dense(self.adjency_size*14, activation='tanh')
        self.reshape3 = layers.Reshape((self.adjency_size, 14))
    
    def call(self, x):
        x,a = x
        x1 = self.conv1([x,a])
        n1 = self.norm(x1)
        x2 = self.flat1(n1)
        d1 = self.drop1(x2)
        x3 = self.dense1(d1)
        z_mean = self.z_mean(x3)
        z_log_var = self.z_log_var(x3)
        z = Sampling()([z_mean, z_log_var])
        
        x4 = self.dense2(z)        
        decodedA = self.reshape2(x4)
        
        x5= self.dense3(z)        
        decodedX = self.reshape3(x5)
        return decodedX, decodedA
    
    def train_step(self, data,verbose=False):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x,a = data

        with tf.GradientTape() as tape:
            x_pred, a_pred = self(data, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = loss_f([x,a], [x_pred, a_pred],verbose)
        if verbose:
            print(loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(x, x_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    