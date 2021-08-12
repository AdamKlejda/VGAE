from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow import keras
from custom_layers import *
class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(Model):
    def __init__(self, latent_dim,n_hidden,num_conv, num_dense):
        super(Encoder, self).__init__()
        self.num_conv = num_conv
        self.num_dense = num_dense
#         self.conv1 = GATConv_layer_relu(n_hidden)    
#         self.conv2 = GATConv_layer_relu(int(n_hidden/2))   Z
#         self.conv3 = GATConv_layer_relu(int(n_hidden/2))
        self.conv1 = Conv_layer_relu(n_hidden)    
        self.conv2 = Conv_layer_relu(n_hidden)   
        self.conv3 = Conv_layer_relu(n_hidden) 
        self.conv4 = Conv_layer_relu(int(n_hidden/2))    
        self.conv5 = Conv_layer_relu(int(n_hidden/2)) 
        
#         self.conv1 = MPConv_layer_relu()
#         self.conv2 = MPConv_layer_relu()
#         self.conv3 = MPConv_layer_relu()
#         self.conv4 = MPConv_layer_relu()
#         self.conv5 = MPConv_layer_relu()
        
        self.flat = layers.Flatten()
        
        self.dense1 = Dense_layer_relu(int(n_hidden))
        self.dense2 = Dense_layer_relu(int(n_hidden))
        self.dense3 = Dense_layer_relu(int(n_hidden/2))
        self.dense4 = Dense_layer_relu(int(n_hidden/2))
        self.dense5 = Dense_layer_relu(int(n_hidden/2))
        
        self.denset = Dense_layer_tanh(int(latent_dim*4))
        
        self.z_mean = Dense_layer_relu(latent_dim,dropout=0.0)        
        self.z_log_var = Dense_layer_relu(latent_dim,dropout=0.0)
        
    def call(self,x):
        x,a = x
        
#         a = tf.sparse.from_dense(a[0])
        x1 = self.conv1([x,a])

        if self.num_conv > 0:
            x1 = self.conv2([x1,a])
        if self.num_conv > 1:
            x1 = self.conv3([x1,a])
        if self.num_conv > 2:
            x1 = self.conv4([x1,a])
        if self.num_conv > 3:
            x1 = self.conv5([x1,a])
        
#         x1 = GlobalAvgPool()(x1)
        x1 = self.flat(x1)
        if self.num_dense>0:
            x1 = self.dense1(x1)
        if self.num_dense>1:
            x1 = self.dense2(x1)
        if self.num_dense>2:
            x1 = self.dense3(x1)
        if self.num_dense>3:
            x1 = self.dense4(x1)
        if self.num_dense>4:
            x1 = self.dense5(x1)        
        
        x1 = self.denset(x1)
        
        z_mean = self.z_mean(x1)
        z_log_var = self.z_log_var(x1)

        z = Sampling()([z_mean, z_log_var])

        return z_mean, z_log_var, z
    


    
class DecoderA(Model):
    def __init__(self, adjency_size,latent_dim,num_dense):
        super(DecoderA, self).__init__()
        self.adjency_size = adjency_size
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        
        self.adense1 = Dense_layer_relu(self.latent_dim*2)  
        
        self.adense2 = Dense_layer_relu(self.latent_dim*4)
        
        self.adense3 = Dense_layer_relu(self.latent_dim*8)
        
        self.adense4 = Dense_layer_relu(self.latent_dim*8)

        self.adense_end = layers.Dense(self.adjency_size*self.adjency_size,
                                    kernel_initializer=initializers.GlorotUniform(seed=None))
        self.adrop_end = layers.Dropout(.2)
        
        self.reshape = layers.Reshape((self.adjency_size, self.adjency_size))
    
    def call(self,z):
        
        da = self.adense1(z)
        if self.num_dense > 1:
            da = self.adense2(da)
        if self.num_dense > 2:
            da = self.adense3(da)
        if self.num_dense > 3:
            da = self.adense4(da)
        
        da = self.adense_end(da)
        da = tf.keras.activations.sigmoid(da)        
        decodedA = self.reshape(da)

        return decodedA

class DecoderX(Model):
    def __init__(self,latent_dim, adjency_size,num_features,num_conv,num_dense):        
        super(DecoderX, self).__init__()
        self.adjency_size = adjency_size
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.num_conv = num_conv
        self.num_dense = num_dense

        self.xdense1  = Dense_layer_relu(self.adjency_size*self.latent_dim)

        self.xreshape1 = layers.Reshape((self.adjency_size, self.latent_dim))
        
        self.xconv1 = Conv_layer_relu(16)        
        self.xconv2 = Conv_layer_relu(32)
#         self.xconv1 = MPConv_layer_relu()
#         self.xconv2 = MPConv_layer_relu()

        
        self.xflat1 = layers.Flatten()

        self.xdense2 = Dense_layer_relu(32)        
        self.xdense3 = Dense_layer_relu(32)
        self.xdense4 = Dense_layer_relu(64)
        self.xdense5 = Dense_layer_relu(128)
        self.xdense6 = Dense_layer_relu(256)
        
        self.xdense_end  = Dense_layer_relu(self.adjency_size*self.num_features)

        
        self.xreshape2 = layers.Reshape((self.adjency_size, self.num_features))
        
    def call(self,x):
        z,decodedA = x
#         decodedA = tf.sparse.from_dense(decodedA[0])
        
        dx = self.xdense1(z)

        if self.num_conv > 0:
            dx = self.xreshape1(dx)
            dx = self.xconv1([dx,decodedA]) 
        
        if self.num_conv > 1:
            dx = self.xconv2([dx,decodedA])

        if self.num_conv > 0: 
            dx = self.xflat1(dx)

        
        if self.num_dense>1:
            dx = self.xdense2(dx)
        if self.num_dense>2:      
            dx = self.xdense3(dx)
        if self.num_dense>3:
            dx = self.xdense4(dx)
        if self.num_dense>4:
            dx = self.xdense5(dx)
        if self.num_dense>5:
            dx = self.xdense6(dx)
        
        
        dx = self.xdense_end(dx)
        
        decodedX = self.xreshape2(dx)-1
        return decodedX

class VGAE(keras.Model):
    def __init__(self, encoder, decoderA, decoderX, **kwargs):
        super(VGAE, self).__init__(**kwargs)
        
        self.encoder = encoder
        self.decoderA = decoderA
        self.decoderX = decoderX
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.reconstruction_loss_trackerX = keras.metrics.Mean(
            name="reconstruction_lossX"
        )
        self.reconstruction_loss_trackerA = keras.metrics.Mean(
            name="reconstruction_lossA"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def call(self,data):
        z_mean, z_log_var, z = self.encoder(data)

        reconstructionA = self.decoderA(z)
        reconstructionX = self.decoderX([z,reconstructionA])
        return reconstructionX, reconstructionA 
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x_true,a_true = data

            z_mean, z_log_var, z = self.encoder(data)

            reconstructionA = self.decoderA(z)

            reconstructionX = self.decoderX([z,reconstructionA])
            reconstruction_lossA = tf.reduce_mean(
                tf.reduce_sum(
                    tf.losses.mean_squared_error(a_true, reconstructionA), axis=(1)
                )
            )
            reconstruction_lossX = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(x_true, reconstructionX), axis=(1)
                )
            )
            reconstruction_loss = reconstruction_lossA + reconstruction_lossX
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.reconstruction_loss_trackerA.update_state(reconstruction_lossA)
        self.reconstruction_loss_trackerX.update_state(reconstruction_lossX)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "reconstruction_lossA": self.reconstruction_loss_trackerA.result(),
            "reconstruction_lossX": self.reconstruction_loss_trackerX.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
