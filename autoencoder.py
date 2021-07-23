from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from spektral.layers import GCNConv
import tensorflow as tf


def loss_f(y_true, y_pred,adjency_size,z_log_var,z_mean,verbose=False):
    loss = 0
    x_true, a_true = y_true
    x_pred, a_pred = y_pred

    if verbose:
        print(x_true[0].numpy().astype(np.float64))
        print(x_pred[0].numpy().astype(np.float64))
    
        print(a_true[0].numpy().astype(np.float64))
        print(a_pred[0].numpy().astype(np.float64))
        
    loss += tf.reduce_mean(tf.abs(tf.cast(tf.squeeze(x_true),tf.float32) - tf.cast(tf.squeeze(x_pred),tf.float32)))
    loss += tf.reduce_mean(tf.square(tf.cast(tf.squeeze(a_true),tf.float32) - tf.cast(tf.squeeze(a_pred),tf.float32)))
    
    log_lik = loss
    kl = (0.5 / adjency_size) * tf.reduce_mean(tf.reduce_sum(1 + 2 * z_log_var - tf.square(z_mean) - tf.square(tf.exp(z_log_var)), 1))
    loss-=kl
    return loss
    
class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Autoencoder(Model):
    def __init__(self, latent_dim,n_hidden,n_samples,adjency_size,num_features):
        super(Autoencoder, self).__init__()
        self.n_samples= n_samples
        self.latent_dim = latent_dim
        self.adjency_size = adjency_size
        self.num_features = num_features
#         encoder
        self.conv1 = GCNConv(n_hidden, activation='relu')
        self.flat1 = layers.Flatten()
        self.drop1 = layers.Dropout(.2)
        
        self.conv2 = GCNConv(n_hidden, activation='relu')
        self.drop2 = layers.Dropout(.2)
        self.norm = layers.BatchNormalization()

        self.z_mean = layers.Dense(latent_dim,name="z_mean")
        self.z_log_var = layers.Dense(latent_dim,name="z_log_var")
        
        
        # decoder A 
        self.adense1 = layers.Dense(64, activation='relu')
        self.adrop1 = layers.Dropout(.2)
        self.adense2 = layers.Dense(self.adjency_size*self.adjency_size, activation='sigmoid')
        self.reshape2 = layers.Reshape((self.adjency_size, self.adjency_size))
        
        # decoder X
#         self.conv3 = GCNConv(n_hidden, activation='tanh')
        self.xdense1  = layers.Dense(self.adjency_size*self.latent_dim, activation='relu')
        self.xreshape1 = layers.Reshape((self.adjency_size, self.latent_dim))
        self.xconv1 = GCNConv(24, activation='relu')
        self.xflat1 = layers.Flatten()
        self.xdrop1 = layers.Dropout(.2)
        self.xdense2  = layers.Dense(self.adjency_size*self.num_features, activation='tanh')
        self.xreshape2 = layers.Reshape((self.adjency_size, self.num_features))
    
    def call(self, x):
        x,a = x
        x1 = self.conv1([x,a])
        x1 = self.norm(x1)
        x1 = self.conv2([x1,a])
        x1 = self.flat1(x1)
        x1 = self.drop1(x1)
#         print(d1.shape)
#         z = x1
        self.z_mean_v = self.z_mean(x1)
        self.z_log_var_v = self.z_log_var(x1)
        z = Sampling()([self.z_mean_v, self.z_log_var_v])
        
        da = self.adense1(z)
        da = self.adrop1(da)
        da = self.adense2(da)
        decodedA = self.reshape2(da)
        
        dx = self.xdense1(z)
        dx = self.xreshape1(dx)
        dx = self.xconv1([dx,decodedA])
        dx = self.xdrop1(dx)
        dx = self.xflat1(dx)
        dx = self.xdense2(dx)
        decodedX = self.xreshape2(dx)
        return decodedX, decodedA
    
    def train_step(self, data,verbose=False):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x,a = data
        
        with tf.GradientTape() as tape:
            x_pred, a_pred = self(data, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = loss_f([x,a], [x_pred, a_pred],self.adjency_size,self.z_log_var_v,self.z_mean_v,verbose)
            
        if verbose:
            print(loss)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state([x,a], [x_pred, a_pred])
        # Return a dict mapping metric names to current value
        return loss
    
