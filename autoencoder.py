from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
from spektral.layers import GCNConv,GATConv
import tensorflow as tf
from tensorflow import keras

    
class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(Model):
    def __init__(self, latent_dim,n_hidden):
        super(Encoder, self).__init__()
        
        self.norm1 = layers.BatchNormalization()        
        self.conv1 = GATConv(n_hidden, activation='relu')
        self.drop1 = layers.Dropout(.2)
        
        self.norm2 = layers.BatchNormalization()
        self.conv2 = GCNConv(int(n_hidden/2), activation='relu')
        self.drop2 = layers.Dropout(.2)
        
        self.norm3 = layers.BatchNormalization()
        self.conv3 = GCNConv(int(n_hidden/4), activation='relu')
        self.drop3 = layers.Dropout(.2)
        
        self.flat = layers.Flatten()


        self.z_mean = layers.Dense(latent_dim,name="z_mean", activation='sigmoid')
        self.z_log_var = layers.Dense(latent_dim,name="z_log_var")
        
    def call(self,x):
        x,a = x
        x1 = self.norm1(x)
        x1 = self.conv1([x1,a])        
        x1 = self.drop1(x1)
        
        x1 = self.norm2(x1)
        x1 = self.conv2([x1,a])
        x1 = self.drop2(x1)
        
        x1 = self.norm3(x1)
        x1 = self.conv3([x1,a])
        x1 = self.drop3(x1)
        
        x1 = self.flat(x1)

        z_mean = self.z_mean(x1)
        z_log_var = self.z_log_var(x1)
        z = Sampling()([z_mean, z_log_var])
        return z_mean, z_log_var, z
    
# encoder = Encoder(latent_dim,n_hidden)


class DecoderA(Model):
    def __init__(self, adjency_size):
        super(DecoderA, self).__init__()
        self.adjency_size = adjency_size
        
        self.anorm1 = layers.BatchNormalization()
        self.adense1 = layers.Dense(128, activation='relu')
        self.adrop1 = layers.Dropout(.2)
        
        self.anorm2 = layers.BatchNormalization()
        self.adense2 = layers.Dense(self.adjency_size*self.adjency_size, activation='sigmoid')
        self.adrop2 = layers.Dropout(.2)
        
        self.reshape2 = layers.Reshape((self.adjency_size, self.adjency_size))
    
    def call(self,z):
        
        da = self.anorm1(z)
        da = self.adense1(da)
        da = self.adrop1(da)
        
        da = self.anorm2(da)
        da = self.adense2(da)
        da = self.adrop2(da)
        
        decodedA = self.reshape2(da)
        return decodedA
        
class DecoderX(Model):
    def __init__(self,latent_dim, adjency_size,num_features):        
        super(DecoderX, self).__init__()
        self.adjency_size = adjency_size
        self.num_features = num_features
        self.latent_dim = latent_dim
        
        self.xnorm1 = layers.BatchNormalization()
        self.xdense1  = layers.Dense(self.adjency_size*self.latent_dim, activation='relu')
        self.xdrop1 = layers.Dropout(.2)

        self.xreshape1 = layers.Reshape((self.adjency_size, self.latent_dim))
        
        self.xnorm2 = layers.BatchNormalization()
        self.xconv1 = GCNConv(64, activation='relu')
        self.xdrop2 = layers.Dropout(.2)

        self.xflat1 = layers.Flatten()

        
        self.xdense2  = layers.Dense(self.adjency_size*self.num_features, activation='tanh')
        self.xdrop3 = layers.Dropout(.2)
        self.xreshape2 = layers.Reshape((self.adjency_size, self.num_features))
        
    def call(self,z,decodedA):
        
        dx = self.xnorm1(z)
        dx = self.xdense1(dx)
        dx = self.xdrop1(dx)
        
        dx = self.xreshape1(dx)
        
        dx = self.xnorm2(dx)
        dx = self.xconv1([dx,decodedA])
        dx = self.xdrop2(dx)
        dx = self.xflat1(dx)
        
        dx = self.xdense2(dx)
        dx = self.xdrop3(dx)
        decodedX = self.xreshape2(dx)
        return decodedX
    
# decoderA = DecoderA(ADJ_SIZE)
# decoderX = DecoderX(latent_dim,ADJ_SIZE,NUM_FEATURES)


class VGAE(keras.Model):
    def __init__(self, encoder, decoderA, decoderX, **kwargs):
        super(VGAE, self).__init__(**kwargs)
        
        self.huber = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
        self.encoder = encoder
        self.decoderA = decoderA
        self.decoderX = decoderX
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
            x_true,a_true = data
            z_mean, z_log_var, z = self.encoder(data)
            reconstructionA = self.decoderA(z)
            reconstructionX = self.decoderX(z,a_true)
#             reconstruction_lossA = self.huber(a_true, reconstructionA)
            reconstruction_lossA = tf.reduce_sum(
                tf.reduce_sum(
#                     keras.losses.mean_squared_error(a_true, reconstructionA), axis=(1)
                    tf.losses.mean_squared_logarithmic_error(a_true, reconstructionA), axis=(1)
                )
            )
#             reconstruction_lossX = self.huber(x_true, reconstructionX)
            reconstruction_lossX = tf.reduce_sum(
                tf.reduce_sum(
                    keras.losses.mean_absolute_error(x_true, reconstructionX), axis=(1)
#                     tf.losses.mean_absolute_percentage_error(x_true, reconstructionX), axis=(1)
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
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
# autoencoder = VGAE(encoder,decoderA,decoderX)
# autoencoder.compile(optimizer=keras.optimizers.Adam())

# losses_all = []
