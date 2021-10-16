from spektral.layers import GCNConv, GlobalSumPool,MessagePassing,GATConv,ARMAConv,ECCConv,GCSConv
import tensorflow.keras.layers as layers
from tensorflow.keras import initializers
import tensorflow as tf
from enum import Enum


class ConvTypes(Enum):
    GCNConv = "gcnconv"
    ARMAConv = "armaconv"
    ECCConv = "eccconv"
    GATConv = "gatconv"
    GCSConv = "gcsconv"
    def __str__(self):
        return self.name

class Conv_layer_relu(layers.Layer):
    def __init__(self,n_hidden,dropout=0.2):
        super(Conv_layer_relu,self).__init__()
        self.conv = GCNConv(n_hidden,
                            kernel_initializer=initializers.he_uniform(seed=None),
                            #kernel_regularizer="l1"
                           )
        self.norm = layers.LayerNormalization() #BatchNormalization       
        self.act  = layers.ReLU()         
        self.drop = layers.Dropout(dropout)
    def call(self, inputs):
        x,a = inputs
        x1 = self.conv([x,a])
        # x1 = self.norm(x1,training)
        x1 = self.norm(x1)
        x1 = self.act(x1) 
        x1 = self.drop(x1)
        return x1
    
class ARMAConv_layer_relu(layers.Layer):
    def __init__(self,n_hidden,dropout=0.2):
        super(ARMAConv_layer_relu,self).__init__()
        self.conv = ARMAConv(n_hidden,
                            kernel_initializer=initializers.he_uniform(seed=None),
                            #kernel_regularizer="l1"
                           )
        self.norm = layers.LayerNormalization()        
        self.act  = layers.ReLU()         
        self.drop = layers.Dropout(dropout)
    def call(self, inputs):
        x,a = inputs
        x1 = self.conv([x,a])
        # x1 = self.norm(x1,training)
        x1 = self.norm(x1)
        x1 = self.act(x1) 
        x1 = self.drop(x1)
        return x1

class ECCConv_layer_relu(layers.Layer):
    def __init__(self,n_hidden,dropout=0.2):
        super(ECCConv_layer_relu,self).__init__()
        self.conv = ECCConv(n_hidden,
                            kernel_initializer=initializers.he_uniform(seed=None),
                            #kernel_regularizer="l1"
                           )
        self.norm = layers.LayerNormalization()        
        self.act  = layers.ReLU()         
        self.drop = layers.Dropout(dropout)
    def call(self, inputs):
        x,a = inputs
        x1 = self.conv([x,a])
        # x1 = self.norm(x1,training)
        x1 = self.norm(x1)
        x1 = self.act(x1) 
        x1 = self.drop(x1)
        return x1

class GCSConv_layer_relu(layers.Layer):
    def __init__(self,n_hidden,dropout=0.2):
        super(GCSConv_layer_relu,self).__init__()
        self.conv = GCSConv(n_hidden,
                            kernel_initializer=initializers.he_uniform(seed=None),
                            #kernel_regularizer="l1"
                           )
        self.norm = layers.LayerNormalization()        
        self.act  = layers.ReLU()         
        self.drop = layers.Dropout(dropout)
    def call(self, inputs):
        x,a = inputs
        x1 = self.conv([x,a])
        # x1 = self.norm(x1,training)
        x1 = self.norm(x1)
        x1 = self.act(x1) 
        x1 = self.drop(x1)
        return x1

class GATConv_layer_relu(layers.Layer):
    def __init__(self,channels,dropout=0.2):
        super(GATConv_layer_relu,self).__init__()
        self.conv = GATConv(channels,
                            kernel_initializer=initializers.he_uniform(seed=None),
                            #kernel_regularizer="l1"
                           )
        self.norm = layers.LayerNormalization()        
        self.act = layers.ReLU()         
        self.drop = layers.Dropout(dropout)
    def call(self, inputs):
        x,a = inputs
        x1 = self.conv([x,a])
        # x1 = self.norm(x1,training)
        x1 = self.norm(x1)
        x1 = self.act(x1) 
        x1 = self.drop(x1)
        return x1


    
class Dense_layer_relu(layers.Layer):
    def __init__(self,n_hidden,dropout=0.2):
        super(Dense_layer_relu,self).__init__()
        self.dense = layers.Dense(n_hidden,
                                  kernel_initializer=initializers.he_uniform(seed=None),
                                  #kernel_regularizer="l1"
                                 )
        self.norm = layers.LayerNormalization()        
        self.act  = layers.ReLU()         
        self.drop = layers.Dropout(dropout)
    
    def call(self, inputs):
        x1 = self.dense(inputs)
        # x1 = self.norm(x1,training)
        x1 = self.norm(x1)
        x1 = self.act(x1) 
        x1 = self.drop(x1)
        return x1
    
class Dense_layer_tanh(layers.Layer):
    def __init__(self,n_hidden,dropout=0.2):
        super(Dense_layer_tanh,self).__init__()
        self.dense = layers.Dense(n_hidden,
                                  kernel_initializer=initializers.GlorotUniform(seed=None),
                                  #kernel_regularizer="l1"
                                  )
        self.norm = layers.LayerNormalization()        
        self.drop = layers.Dropout(dropout)
    
    def call(self, inputs):
        x1 = self.dense(inputs)
#         x1 = self.norm(x1,training)
        x1 = tf.keras.activations.tanh(x1)
#         x1 = self.drop(x1)
        return x1