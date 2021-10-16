import tensorflow as tf
from tensorflow import keras
from custom_layers import *
from utils import FramsManager,gen_f0_from_tensors


class VGAE_l(keras.Model):
    def __init__(self, pathFrams, encoder, decoderA, decoderX, **kwargs):
        super(VGAE_l, self).__init__(**kwargs)
        
        self.FramsManager = FramsManager(pathFrams)
        self.encoder = encoder
        self.decoderA = decoderA
        self.decoderX = decoderX
        
    def call(self,data):
        z_mean, z_log_var, z = self.encoder(data)

        reconstructionA = self.decoderA(z)
        reconstructionX = self.decoderX([z,reconstructionA])
        return reconstructionX, reconstructionA 
    
    def test_step(self, data):        
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

        gen_list = gen_f0_from_tensors(reconstructionX,reconstructionA)
        c_wrong_joints = self.FramsManager.count_wrong_joints(gen_list)
        reconstruction_loss = (reconstruction_lossA*1000) + reconstruction_lossX + c_wrong_joints * 100
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
    
        return total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,kl_loss

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

            gen_list = gen_f0_from_tensors(reconstructionX,reconstructionA)
            c_wrong_joints = self.FramsManager.count_wrong_joints(gen_list)
            reconstruction_loss = (reconstruction_lossA*1000) + reconstruction_lossX + (c_wrong_joints * 100)
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_lossA": reconstruction_lossA,
            "reconstruction_lossX": reconstruction_lossX,
            "kl_loss": kl_loss,
        }

class GAE_l(keras.Model):
    def __init__(self,pathFrams, encoder, decoderA, decoderX, **kwargs):
        super(GAE_l, self).__init__(**kwargs)
        self.FramsManager = FramsManager(pathFrams)
        self.encoder = encoder
        self.decoderA = decoderA
        self.decoderX = decoderX


    def call(self,data):
        z = self.encoder(data)

        reconstructionA = self.decoderA(z)
        reconstructionX = self.decoderX([z,reconstructionA])
        return reconstructionX, reconstructionA 
    
    def test_step(self, data):        
        x_true,a_true = data

        z = self.encoder(data)

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
        gen_list = gen_f0_from_tensors(reconstructionX,reconstructionA)
        c_wrong_joints = self.FramsManager.count_wrong_joints(gen_list)
        reconstruction_loss = (reconstruction_lossA*1000) + reconstruction_lossX + c_wrong_joints * 100
        
        total_loss = reconstruction_loss
    
        return total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x_true,a_true = data

            z = self.encoder(data)

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
            gen_list = gen_f0_from_tensors(reconstructionX,reconstructionA)
            c_wrong_joints = self.FramsManager.count_wrong_joints(gen_list)
            reconstruction_loss = (reconstruction_lossA*1000) + reconstruction_lossX + c_wrong_joints * 100
            
            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_lossA": reconstruction_lossA,
            "reconstruction_lossX": reconstruction_lossX
        }
