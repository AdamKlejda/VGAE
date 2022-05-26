import re
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from tensorflow import keras
from GAE.architecture.base.custom_layers import *
from GAE.architecture.base.weights import Weights

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs 
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
class EncoderVGAE(Model):
    def __init__(self, latent_dim,n_hidden,num_conv, num_dense,convtype=ConvTypes.GCNConv):
        super(EncoderVGAE, self).__init__()
        self.num_conv = num_conv
        self.num_dense = num_dense
        self.convtype = convtype

        self.conv1 = Conv_layers_relu(n_hidden,convtype)    
        self.conv2 = Conv_layers_relu(n_hidden,convtype)   
        self.conv3 = Conv_layers_relu(n_hidden,convtype) 
        self.conv4 = Conv_layers_relu(int(n_hidden/2),convtype)    
        self.conv5 = Conv_layers_relu(int(n_hidden/2),convtype)
             
        self.flat = layers.Flatten()
        
        self.dense1 = Dense_layer_relu(int(n_hidden))
        self.dense2 = Dense_layer_relu(int(n_hidden))
        self.dense3 = Dense_layer_relu(int(n_hidden/2))
        self.dense4 = Dense_layer_relu(int(n_hidden/2))
        self.dense5 = Dense_layer_relu(int(n_hidden/2))
        
        self.denset = Dense_layer_tanh(int(latent_dim*4))
        
        self.z_mean = Dense_layer_relu(latent_dim,dropout=0.0)        
        self.z_log_var = Dense_layer_relu(latent_dim,dropout=0.0)
    
        
    def call(self,x,training=False):
        x,a = x

        x1 = self.conv1([x,a])
        if self.num_conv > 0:
            x1 = self.conv2([x1,a])
        if self.num_conv > 1:
            x1 = self.conv3([x1,a])  
        if self.num_conv > 2:
            x1 = self.conv4([x1,a])
        if self.num_conv > 3:
            x1 = self.conv5([x1,a]) 
        
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

class EncoderGAE(Model):
    def __init__(self, latent_dim,n_hidden,num_conv, num_dense,convtype=ConvTypes.GCNConv):
        super(EncoderGAE, self).__init__()
        self.num_conv = num_conv
        self.num_dense = num_dense
        self.convtype = convtype

        self.conv1 = Conv_layers_relu(n_hidden,convtype)    
        self.conv2 = Conv_layers_relu(n_hidden,convtype)   
        self.conv3 = Conv_layers_relu(n_hidden,convtype) 
        self.conv4 = Conv_layers_relu(int(n_hidden/2),convtype)    
        self.conv5 = Conv_layers_relu(int(n_hidden/2),convtype)       
           
        self.flat = layers.Flatten()
        
        self.dense1 = Dense_layer_relu(int(n_hidden))
        self.dense2 = Dense_layer_relu(int(n_hidden))
        self.dense3 = Dense_layer_relu(int(n_hidden/2))
        self.dense4 = Dense_layer_relu(int(n_hidden/2))
        self.dense5 = Dense_layer_relu(int(n_hidden/2))
        
        self.denset = Dense_layer_tanh(int(latent_dim*4))
        
        self.z = Dense_layer_relu(latent_dim,dropout=0.0)
        
    def call(self,x,training=False):
        x,a = x
        x1 = self.conv1([x,a])    

        if self.num_conv > 0:
            x1 = self.conv2([x1,a])
        if self.num_conv > 1:
            x1 = self.conv3([x1,a])
        if self.num_conv > 2:
            x1 = self.conv4([x1,a])
        if self.num_conv > 3:
            x1 = self.conv5([x1,a]) 
        
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
        
        z = self.z(x1)
        return z    

    
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
    
    def call(self,z,training=False):
        
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
    def __init__(self,latent_dim, adjency_size,num_features,num_conv,num_dense,convtype=ConvTypes.GCNConv):        
        super(DecoderX, self).__init__()
        self.adjency_size = adjency_size
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.num_conv = num_conv
        self.num_dense = num_dense
        self.convtype = convtype

        self.xdense1  = Dense_layer_relu(self.adjency_size*self.latent_dim)
        self.xreshape1 = layers.Reshape((self.adjency_size, self.latent_dim))    

        self.xconv1 = Conv_layers_relu(latent_dim,convtype)    
        self.xconv2 = Conv_layers_relu(latent_dim*2,convtype)  
        
        self.xflat1 = layers.Flatten()

        self.xdense2 = Dense_layer_relu(latent_dim*2)        
        self.xdense3 = Dense_layer_relu(latent_dim*2)
        self.xdense4 = Dense_layer_relu(latent_dim*2)
        self.xdense5 = Dense_layer_relu(latent_dim*3)
        self.xdense6 = Dense_layer_relu(latent_dim*3)
        
        self.xdense_end  = Dense_layer_leaky_relu(self.adjency_size*self.num_features)

        
        self.xreshape2 = layers.Reshape((self.adjency_size, self.num_features))
        
    def call(self,x,training=False):
        z,decodedA = x
        
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
        
        decodedX = self.xreshape2(dx)
        return decodedX

class VGAE(keras.Model,Weights):
    def __init__(self, encoder, decoderA, decoderX, custom_loss=None, **kwargs):
        super(VGAE, self).__init__(**kwargs)
        
        self.encoder = encoder
        self.decoderA = decoderA
        self.decoderX = decoderX
        self.custom_loss = custom_loss

    def call(self, data, training=False):
        x_true, a_true, y_true = data
        z_mean, z_log_var, z = self.encoder.call([x_true,a_true])

        reconstructionA = self.decoderA.call(z)
        reconstructionX = self.decoderX.call([z,reconstructionA])
        return reconstructionX, reconstructionA 
    
    def test_step(self, data):
        
        x_true, a_true, y_true = data
        z_mean, z_log_var, z = self.encoder.call([x_true,a_true])

        reconstructionA = self.decoderA.call(z)

        reconstructionX = self.decoderX.call([z,reconstructionA])
        reconstruction_lossA = tf.reduce_mean(
            tf.reduce_sum(
                tf.losses.mean_squared_error(a_true, reconstructionA), axis=(1)
            )
        )
        reconstruction_lossX = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(x_true[:,:,:-1], reconstructionX[:,:,:-1]), axis=(1)
            )
        )
        reconstruction_lossMask = tf.reduce_mean(
            tf.reduce_sum(
                [keras.losses.mean_squared_error(x_true[:,:,-1:], reconstructionX[:,:,-1:])], axis=(1)
            )
        )

        reconstruction_loss = (reconstruction_lossA*self.weight_loss_A) + reconstruction_lossX*self.weight_loss_X + (reconstruction_lossMask * self.weight_mask_loss)
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    
        custom_loss_v = 0
        if self.custom_loss is not None:
            custom_loss_v = self.custom_loss([x_true,a_true,y_true],[reconstructionX,reconstructionA],z)

        total_loss = reconstruction_loss + kl_loss + custom_loss_v*self.weight_custom_loss
    
        return total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,reconstruction_lossMask ,kl_loss

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            x_true,a_true,y_true = data

            z_mean, z_log_var, z = self.encoder.call([x_true,a_true])

            reconstructionA = self.decoderA.call(z)

            reconstructionX = self.decoderX.call([z,reconstructionA])
            reconstruction_lossA = tf.reduce_mean(
                tf.reduce_sum(
                    tf.losses.mean_squared_error(a_true, reconstructionA), axis=(1)
                )
            )
            reconstruction_lossX = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(x_true[:,:,:-1], reconstructionX[:,:,:-1]), axis=(1)
                )
            )
            reconstruction_lossMask = tf.reduce_mean(
                tf.reduce_sum(
                    [keras.losses.mean_squared_error(x_true[:,:,-1:], reconstructionX[:,:,-1:])], axis=(1)
                )
            )
            reconstruction_loss = (reconstruction_lossA*self.weight_loss_A) + reconstruction_lossX*self.weight_loss_X + (reconstruction_lossMask * self.weight_mask_loss)
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
            custom_loss_v = 0
            if self.custom_loss is not None:
                custom_loss_v = self.custom_loss([x_true,a_true,y_true],[reconstructionX,reconstructionA],z)

            total_loss = reconstruction_loss + kl_loss + custom_loss_v*self.weight_custom_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_lossA": reconstruction_lossA,
            "reconstruction_lossX": reconstruction_lossX,
            "reconstruction_lossMask": reconstruction_lossMask,
            "kl_loss": kl_loss,
        }

class GAE(keras.Model,Weights):
    def __init__(self, encoder, decoderA, decoderX, custom_loss=None, **kwargs):
        super(GAE, self).__init__(**kwargs)
        
        self.encoder = encoder
        self.decoderA = decoderA
        self.decoderX = decoderX
        self.custom_loss = custom_loss

    def call(self,data):
        x_true,a_true,y_true = data
        z = self.encoder.call([x_true,a_true])

        reconstructionA = self.decoderA.call(z)
        reconstructionX = self.decoderX.call([z,reconstructionA])
        return reconstructionX, reconstructionA 
    
    def test_step(self, data):           
        x_true,a_true,y_true = data

        z = self.encoder.call([x_true,a_true])

        reconstructionA = self.decoderA.call(z)

        reconstructionX = self.decoderX.call([z,reconstructionA])
        reconstruction_lossA = tf.reduce_mean(
            tf.reduce_sum(
                tf.losses.mean_squared_error(a_true, reconstructionA), axis=(1)
            )
        )
        reconstruction_lossX = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(x_true[:,:,:-1], reconstructionX[:,:,:-1]), axis=(1)
            )
        )
        reconstruction_lossMask = tf.reduce_mean(
            tf.reduce_sum(
                [keras.losses.mean_squared_error(x_true[:,:,-1:], reconstructionX[:,:,-1:])], axis=(1)
            )
        )
        reconstruction_loss = (reconstruction_lossA*self.weight_loss_A) + reconstruction_lossX*self.weight_loss_X + (reconstruction_lossMask * self.weight_mask_loss)
        
        custom_loss_v = 0
        if self.custom_loss is not None:
            custom_loss_v = self.custom_loss([x_true,a_true,y_true],[reconstructionX,reconstructionA],z)

        total_loss = reconstruction_loss + custom_loss_v*self.weight_custom_loss
    
        return total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,reconstruction_lossMask

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            x_true,a_true,y_true = data

            z = self.encoder.call([x_true,a_true])

            reconstructionA = self.decoderA.call(z)

            reconstructionX = self.decoderX.call([z,reconstructionA])
            reconstruction_lossA = tf.reduce_mean(
                tf.reduce_sum(
                    tf.losses.mean_squared_error(a_true, reconstructionA), axis=(1)
                )
            )
            reconstruction_lossX = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(x_true[:,:,:-1], reconstructionX[:,:,:-1]), axis=(1)
                )
            )
            reconstruction_lossMask = tf.reduce_mean(
                tf.reduce_sum(
                    [keras.losses.mean_squared_error(x_true[:,:,-1:], reconstructionX[:,:,-1:])], axis=(1)
                )
            )
            reconstruction_loss = (reconstruction_lossA*self.weight_loss_A) + reconstruction_lossX*self.weight_loss_X + (reconstruction_lossMask * self.weight_mask_loss)
            
            custom_loss_v = 0
            if self.custom_loss is not None:
                custom_loss_v = self.custom_loss([x_true,a_true,y_true],[reconstructionX,reconstructionA],z)

            total_loss = reconstruction_loss + custom_loss_v*self.weight_custom_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_lossA": reconstruction_lossA,
            "reconstruction_lossX": reconstruction_lossX,
            "reconstruction_lossMask": reconstruction_lossMask,
        }
