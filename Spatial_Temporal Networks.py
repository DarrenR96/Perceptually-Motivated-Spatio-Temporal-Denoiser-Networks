import numpy as np
import pandas as pd 
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class CNNReluBN(layers.Layer):
    def __init__(self, out_channels=96 ,ksize=(3,3), padding='same', dropout=False, dropoutVal = 0.35):
        super(CNNReluBN, self).__init__()
        self.dropoutFlag = dropout
        self.out_channels = out_channels
        self.k_size = ksize
        self.padding = padding
        self.conv = layers.Conv2D(self.out_channels, self.k_size, padding=self.padding)
        self.bn = layers.BatchNormalization()
        if self.dropoutFlag:
            self.dropout = layers.SpatialDropout2D(dropoutVal)
        self.relu = layers.ReLU()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_channels': self.out_channels,
            'ksize': self.k_size,
            'padding': self.padding,
        })
        return config
    
    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.dropoutFlag:
            x = self.dropout(x, training=training)
        x = self.relu(x)
        return x
    
class CNNReluBNMaxPool(layers.Layer):
    def __init__(self, out_channels=96 ,ksize=(3,3), padding='same', data_format='channels_last', dropout=False, dropoutVal = 0.35, pool=(2,2)):
        super(CNNReluBNMaxPool, self).__init__()
        self.dropoutFlag = dropout
        self.out_channels = out_channels
        self.k_size = ksize
        self.padding = padding
        self.data_format = data_format
        self.conv = layers.Conv2D(self.out_channels, self.k_size, padding=self.padding)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(data_format=self.data_format,pool_size=pool)
        if self.dropoutFlag:
            self.dropout = layers.SpatialDropout2D(dropoutVal)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_channels': self.out_channels,
            'ksize': self.k_size,
            'padding': self.padding,
            'data_format': self.data_format
        })
        return config
        
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.bn(x, training=training)
        x = self.maxpool(x)
        if self.dropoutFlag:
            x = self.dropout(x, training=training)
        return x
        
class SpatialDenoiser(keras.Model):

    def __init__(self, num_layers=12, ksize=(3,3)):
        super(SpatialDenoiser, self).__init__()
        self.middle_channels = 96
        self.output_channels = 4
        
        
        # Center CNN
        self.CNNReluBNCollection = []
        for _ in range(num_layers-1):
            self.CNNReluBNCollection.append(CNNReluBN(dropout=True))
        
        # Output CNN
        self.conv_output = layers.Conv2D(self.output_channels, ksize, padding='same')
        self.relu_output = layers.ReLU()
        
        
    def call(self, inputs, noise_map, training=False):
        # Convert (NumBatches,50,50,1) to (NumBatches,25,25,4)
        x = tf.image.extract_patches(inputs, sizes=[1,2,2,1], strides=[1,2,2,1], rates=[1,1,1,1], padding='SAME')
        x = tf.concat([noise_map[:,::2,::2,:], x],-1)
        
        
        for middleConv in self.CNNReluBNCollection:
            x = middleConv(x, training=training)

        x = self.conv_output(x)
        x = self.relu_output(x)
        
        # Convert (NumBatches,25,25,1) to (NumBatches,50,50,1) 
        x = tf.nn.depth_to_space(x, block_size=2)
        
        # Residual Computation
        x = tf.math.subtract(inputs, x)
        
        return x

    def model(self):
        x = keras.Input(shape=(50,50,1))
        noise_map = keras.Input(shape=(50,50,1))
        return keras.Model(inputs=[x,noise_map], outputs=self.call(x,noise_map))

class ImageQualityAssessment(keras.Model):

    def __init__(self, independent_layers_channels=[24,48], concat_layers_channels=[48,48], dense_connections=[256,128,1],ksize=(5,5)):
        super(ImageQualityAssessment, self).__init__()
        self.ConvReluBNCollection_independent = []
        self.ConvReluBNCollection_concat = []
        self.Dense_layers = []
        self.Flatten = layers.Flatten()
        
        # Cascade of shared weights for Spatial Evaluation
        for i in range(len(independent_layers_channels)):
            self.ConvReluBNCollection_independent.append(CNNReluBNMaxPool(padding='same', out_channels=independent_layers_channels[i], ksize=ksize))
        
        self.inputRes = CNNReluBNMaxPool(padding='same', out_channels=48, ksize=(3,3), pool=(4,4))    
            
        # Cascade of weights for Concat Evaluation    
        for i in range(len(concat_layers_channels)):
            self.ConvReluBNCollection_concat.append(CNNReluBN(padding='same', out_channels=concat_layers_channels[i], ksize=ksize))
        
        self.outputRes = CNNReluBN(padding='same', out_channels=concat_layers_channels[-1],ksize=(1,1))
        
        # Dense Layers
        for i in range(len(dense_connections)):
            self.Dense_layers.append(layers.Dense(dense_connections[i], activation='relu'))
        
    def call(self, x_ref ,x_restored, training=False):
        
        # Processing reference and restored on same CNN->BN->RELU blocls
        x_ref_in = x_ref
        for cnn_relu_bn_block in self.ConvReluBNCollection_independent:
            x_ref = cnn_relu_bn_block(x_ref) 
        x_ref_in = self.inputRes(x_ref_in)
        x_ref = x_ref+x_ref_in
        
        x_rest_in = x_restored
        for cnn_relu_bn_block in self.ConvReluBNCollection_independent:
            x_restored = cnn_relu_bn_block(x_restored)
        x_rest_in = self.inputRes(x_rest_in)
        x_restored = x_restored+x_rest_in
         
        # Concat output 
        x = tf.concat([x_ref, x_restored],-1)
        x_in = x
        # CNN->BN->RELU applied on concat output followed by Dense
        for cnn_relu_bn_block in self.ConvReluBNCollection_concat:
            x = cnn_relu_bn_block(x)
        x_in = self.outputRes(x_in)
        x = x + x_in
        x = self.Flatten(x)
        for dense_layer in self.Dense_layers:
            x = dense_layer(x)
            
        return x

    def model(self):
        x_ref = keras.Input(shape=(50,50,1))
        x_restored = keras.Input(shape=(50,50,1))
        return keras.Model(inputs=[x_ref,x_restored], outputs=self.call(x_ref,x_restored))

class TemporalDenoiser(keras.Model):

    def __init__(self, num_layers=6, ksize=(3,3)):
        super(TemporalDenoiser, self).__init__()
        self.output_channels = 4
        
        # Center CNN
        self.CNNReluBNCollection = []
        for _ in range(num_layers-1):
            self.CNNReluBNCollection.append(CNNReluBN())
        
        # Output CNN
        self.conv_output = layers.Conv2D(self.output_channels, ksize, padding='same')
        self.relu_output = layers.ReLU()
        
        
    def call(self,x_t,noise_map, training=True):
        x_tmin1 = x_t[:,0,:,:]
        x_tpl1 = x_t[:,2,:,:]
        x_t = x_t[:,1,:,:]
        # Convert (NumBatches,50,50,1) to (NumBatches,25,25,4)
        x_tmin1_rs = tf.image.extract_patches(x_tmin1, sizes=[1,2,2,1], strides=[1,2,2,1], rates=[1,1,1,1], padding='SAME')
        x_t_rs = tf.image.extract_patches(x_t, sizes=[1,2,2,1], strides=[1,2,2,1], rates=[1,1,1,1], padding='SAME')
        x_tpl1_rs = tf.image.extract_patches(x_tpl1, sizes=[1,2,2,1], strides=[1,2,2,1], rates=[1,1,1,1], padding='SAME')
        x = tf.concat([noise_map[:,::2,::2,:], x_tmin1_rs, x_t_rs, x_tpl1_rs],-1)
        
        # Neural Network Layer        
        for middleConv in self.CNNReluBNCollection:
            x = middleConv(x, training=training)
        x = self.conv_output(x)
        x = self.relu_output(x)
        
        
        # Convert (NumBatches,25,25,1) to (NumBatches,50,50,1) 
        x = tf.nn.depth_to_space(x, block_size=2)

        # Residual Computation
        x = tf.math.subtract(x_t, x)
        
        return x

    def model(self):
        x_t = keras.Input(shape=(3,50,50,1))
        noise_map = keras.Input(shape=(50,50,1))
        return keras.Model(inputs=[x_t,noise_map], outputs=self.call(x_t,noise_map))

class VideoQualityAssessment(keras.Model):

    def __init__(self, independent_spatial_channels=[24,48], independent_temporal_channels=[48,64], concat_layers_channels=[64,64], dense_connections=[256,128,1],ksize=(5,5)):
        super(VideoQualityAssessment, self).__init__()
        self.ConvReluBNColl_spatial_independent = []
        self.ConvReluBNColl_temporal_independent = []
        self.ConvReluBNCollection_concat = []
        self.Dense_layers = []
        self.Flatten = layers.Flatten()
        
        # Cascade of shared weights for Spatial Evaluation
        for i in range(len(independent_spatial_channels)):
            self.ConvReluBNColl_spatial_independent.append(CNNReluBNMaxPool(padding='same', out_channels=independent_spatial_channels[i], ksize=ksize))
        
        self.inputRes = CNNReluBNMaxPool(padding='same', out_channels=48, ksize=(3,3), pool=(4,4))    
        
        # Cascade of shared weights for Temporal Evaluation    
        for i in range(len(independent_temporal_channels)):
            self.ConvReluBNColl_temporal_independent.append(CNNReluBN(padding='same', out_channels=independent_temporal_channels[i], ksize=ksize))
        
        self.middleRes = CNNReluBN(padding='same', out_channels=64,ksize=(1,1))
        
         # Cascade of weights for concat Evaluation    
        for i in range(len(concat_layers_channels)):
            self.ConvReluBNCollection_concat.append(CNNReluBN(padding='same', out_channels=concat_layers_channels[i], ksize=ksize))
        
        self.outputRes = CNNReluBN(padding='same', out_channels=64,ksize=(1,1))
        
        # Dense Layers
        for i in range(len(dense_connections)):
            self.Dense_layers.append(layers.Dense(dense_connections[i], activation='relu'))
        
    def call(self, x_t_min1_ref, x_t_ref, x_t_pl1_ref, x_t_min1_restored, x_t_restored, x_t_pl1_restored, training=False):
        
        x_t_min1_ref_in = x_t_min1_ref
        x_t_ref_in = x_t_ref
        x_t_pl1_ref_in = x_t_pl1_ref
        
        # Processing reference and restored on same spatial CNN->BN->RELU blocks
        for cnn_relu_bn_block in self.ConvReluBNColl_spatial_independent:
            x_t_min1_ref = cnn_relu_bn_block(x_t_min1_ref)
        for cnn_relu_bn_block in self.ConvReluBNColl_spatial_independent:
            x_t_ref = cnn_relu_bn_block(x_t_ref) 
        for cnn_relu_bn_block in self.ConvReluBNColl_spatial_independent:
            x_t_pl1_ref = cnn_relu_bn_block(x_t_pl1_ref) 
            
        x_t_min1_ref_in = self.inputRes(x_t_min1_ref_in)
        x_t_min1_ref = x_t_min1_ref + x_t_min1_ref_in
        
        x_t_ref_in = self.inputRes(x_t_ref_in)
        x_t_ref = x_t_ref + x_t_ref_in
        
        x_t_pl1_ref_in = self.inputRes(x_t_pl1_ref_in)
        x_t_pl1_ref = x_t_pl1_ref + x_t_pl1_ref_in
        
        
        
        x_t_min1_restored_in = x_t_min1_restored
        x_t_restored_in = x_t_restored
        x_t_pl1_restored_in = x_t_pl1_restored
        
        for cnn_relu_bn_block in self.ConvReluBNColl_spatial_independent:
            x_t_min1_restored = cnn_relu_bn_block(x_t_min1_restored)
        for cnn_relu_bn_block in self.ConvReluBNColl_spatial_independent:
            x_t_restored = cnn_relu_bn_block(x_t_restored) 
        for cnn_relu_bn_block in self.ConvReluBNColl_spatial_independent:
            x_t_pl1_restored = cnn_relu_bn_block(x_t_pl1_restored) 
            
        x_t_min1_restored_in = self.inputRes(x_t_min1_restored_in)
        x_t_min1_restored = x_t_min1_restored + x_t_min1_restored_in
        
        x_t_restored_in = self.inputRes(x_t_restored_in)
        x_t_restored = x_t_restored + x_t_restored_in
        
        x_t_pl1_restored_in = self.inputRes(x_t_pl1_restored_in)
        x_t_pl1_restored = x_t_pl1_restored + x_t_pl1_restored_in  
            
        
        # Concat outputs of spatial blocks
        x_ref = tf.concat([x_t_min1_ref, x_t_ref, x_t_pl1_ref],-1)
        x_restored = tf.concat([x_t_min1_restored, x_t_restored, x_t_pl1_restored],-1)
        
        # Processing reference and restored on same temporal CNN->BN->RELU blocks
        x_ref_in = x_ref
        x_rest_in = x_restored
        for cnn_relu_bn_block in self.ConvReluBNColl_temporal_independent:
            x_ref = cnn_relu_bn_block(x_ref) 
        for cnn_relu_bn_block in self.ConvReluBNColl_temporal_independent:
            x_restored = cnn_relu_bn_block(x_restored)
        x_ref_in = self.middleRes(x_ref_in)
        x_ref = x_ref + x_ref_in
        x_rest_in = self.middleRes(x_rest_in)
        x_restored = x_restored + x_rest_in
            
        # Concat outputs of temporal block
        x = tf.concat([x_ref,x_restored],-1)
        
        x_in = x
        # CNN->BN->RELU applied on concat output followed by Dense
        for cnn_relu_bn_block in self.ConvReluBNCollection_concat:
            x = cnn_relu_bn_block(x)
        x_in = self.outputRes(x_in)
        x = x + x_in
        x = self.Flatten(x)
        for dense_layer in self.Dense_layers:
            x = dense_layer(x)
            
        return x

    def model(self):
        x_t_min1_ref = keras.Input(shape=(50,50,1))
        x_t_ref = keras.Input(shape=(50,50,1))
        x_t_pl1_ref = keras.Input(shape=(50,50,1))

        x_t_min1_restored = keras.Input(shape=(50,50,1))
        x_t_restored = keras.Input(shape=(50,50,1))
        x_t_pl1_restored = keras.Input(shape=(50,50,1))

        return keras.Model(inputs=[x_t_min1_ref, x_t_ref, x_t_pl1_ref, x_t_min1_restored, x_t_restored, x_t_pl1_restored], 
                           outputs=self.call(x_t_min1_ref, x_t_ref, x_t_pl1_ref, x_t_min1_restored, x_t_restored, x_t_pl1_restored))
