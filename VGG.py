import tensorflow as tf
import numpy as np
import os 
import math
import cv2
slim = tf.contrib.slim
class VGG16():
    def __init__(self,is_training = True,with_batch_norm=True):
        self.scope = 'vgg_16'
        self.dropout_keep_prob = 0.5
        self.is_training = is_training
        self.padding ='SAME'
        self.skip_fc = True
        self.with_batch_norm = with_batch_norm
        self.features = ['block04', 'block07', 'block08', 'block09', 'block10', 'block11']
        self.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (2, 2)]
        self.anchor_sizes = [ (12., 16.)    ,
                             (16., 32.)     ,
                              (40., 64.)    ,  
                              (80., 128.)   ,
                              (140., 196.)  ,
                              (210., 240., 270.)  ]
        self.anchor_ratios =   [ [2, 0.5]         ,
                               [2, 0.5, 3, 1./3]  ,
                              [2, 0.5, 3, 1./3]   ,
                               [2, 0.5, 3, 1./3]  ,
                               [2, 0.5, 3, 1./3]  ,
                              [2, 0.5] ]
        #self.features = ['block03','block04', 'block07', 'block08', 'block09', 'block10', 'block11']
        #self.feat_shapes = [(75,75),(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        #self.anchor_sizes = [ (),(21., 45.)    ,4
        #                      (45., 99.)    ,
        #                      (99., 153.)   ,  
        #                      (153., 207.)  ,
        #                      (207., 261.)  ,
        #                      (261., 315.)  ]                               
        self.anchor_steps = [8, 16, 32, 64, 100, 300]
        ### Dataset  Specific ###
        self.num_classes = 21
        self.img_shape = [300,300]

        ###Traininig Specification###
        self.batch_size = 16
        
        pass
    def _build(self,X):
        self.X          = X
        with tf.variable_scope(self.scope) as sc:
            self.end_points = {}
            #with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding='SAME',weights_regularizer=slim.l2_regularizer(0.0005)):
            with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding='SAME'):
                with slim.arg_scope([slim.conv2d,slim.fully_connected],
                    activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.0001)):
                    
                    net = X
                    net = slim.repeat(net,2,slim.conv2d,64,[3,3] ,scope='conv1')
                    self.end_points['block01'] = net
                    net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn1')
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool1')
                    
                    net = slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
                    self.end_points['block02'] = net
                    net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn2')           
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool2')
                    
                    net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
                    net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn3')           
                    self.end_points['block03'] = net
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool3')
                    
                    net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv4')
                    net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn4')           
                    self.end_points['block04'] = net                
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool4')
                    
                    net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')
                    net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn5')           
                    self.end_points['block05'] = net
                    #net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                    net = slim.max_pool2d(net,[3,3],[1,1]        ,scope='pool5')
                    
                    if self.skip_fc:
                        net = slim.conv2d(net,1024,[3,3],rate=6  ,scope='conv6')
                        self.end_points['block06'] = net
                        net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                        
                        net = slim.conv2d(net,1024,[1,1]         ,scope='conv7')
                        self.end_points['block07'] = net
                        net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                        
                        
                    with tf.variable_scope('block08'):
                        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                        net = self.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                        net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn8')           
                        self.end_points['block08'] = net
                        net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                        
                    with tf.variable_scope('block09'):
                        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                        net = self.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                        net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn9')           
                        self.end_points['block09'] = net
                        net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                        
                    with tf.variable_scope('block10'):
                        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                        net = self.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                        net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn10')           
                        self.end_points['block10'] = net
                        net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                    
                    with tf.variable_scope('block11'):
                        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                        net = self.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                        net = self.Batch_Normalization(net,is_training=self.is_training,scope='bn11')           
                        self.end_points['block11'] = net
                        net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                    self.logits = net

                    
                
    def pad2d(self,inputs,pad=(0,0),mode ='CONSTANT', scope=None):
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
            net = tf.pad(inputs, paddings, mode='CONSTANT')
            return net
    def Batch_Normalization(self,x,is_training=True,scope=None):
        if not self.with_batch_norm:
            return x
        with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                            scope=scope,updates_collections=None,decay=0.9,center=True,scale=True,zero_debias_moving_mean=True):
            
            net = tf.cond(is_training,
                          lambda: tf.contrib.layers.batch_norm(inputs=x,is_training=is_training,reuse=None),
                          lambda: tf.contrib.layers.batch_norm(inputs=x,is_training=is_training,reuse=True))
            return net
                

                
                
                
                
                

                    
            
            
                