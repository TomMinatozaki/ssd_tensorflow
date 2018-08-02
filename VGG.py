import tensorflow as tf
import numpy as np
import os 
import math
slim = tf.contrib.slim
class VGG16():
    def __init__(self):
        self.scope = 'vgg16'
        self.dropout_keep_prob = 0.5
        self.is_training = True
        self.padding ='SAME'
        self.skip_fc = True
        self.features = ['block04', 'block07', 'block08', 'block09', 'block10', 'block11']
        self.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        self.anchor_sizes = [ (12., 16.)    ,
                             (16., 32.)    ,
                              (32., 64.)   ,  
                              (64., 128.)  ,
                              (128., 196.)  ,
                              (224., 325.)  ]
        self.anchor_ratios =   [ [2, 0.5]         ,
                               [2, 0.5, 3, 1./3],
                              [2, 0.5, 3, 1./3],
                               [2, 0.5, 3, 1./3],
                               [2, 0.5],
                              [2, 0.5] ]
        #self.features = ['block03','block04', 'block07', 'block08', 'block09', 'block10', 'block11']
        #self.feat_shapes = [(75,75),(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        #self.anchor_sizes = [ (),(21., 45.)    ,
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
        with tf.variable_scope(self.scope) as sc:
            self.end_points = {}
            #with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding='SAME',weights_regularizer=slim.l2_regularizer(0.0005)):
            with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding='SAME'):
                with slim.arg_scope([slim.conv2d,slim.fully_connected],activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = X
                    net = slim.repeat(net,2,slim.conv2d,64,[3,3] ,scope='conv1')
                    self.end_points['block01'] = net
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool1')
                    
                    net = slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
                    self.end_points['block02'] = net
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool2')
                    
                    net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
                    self.end_points['block03'] = net
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool3')
                    
                    net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv4')
                    self.end_points['block04'] = net                
                    net = slim.max_pool2d(net,[2,2]              ,scope='pool4')
                    
                    net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')
                    self.end_points['block05'] = net
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
                        self.end_points['block08'] = net
                        
                    with tf.variable_scope('block09'):
                        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                        net = self.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                        self.end_points['block09'] = net
                        
                    with tf.variable_scope('block10'):
                        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                        net = self.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                        self.end_points['block10'] = net
                    
                    with tf.variable_scope('block11'):
                        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                        net = self.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
                        self.end_points['block11'] = net
                    
                    self.logits = net

                    
                
    def pad2d(self,inputs,pad=(0,0),mode ='CONSTANT', scope=None):
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
            net = tf.pad(inputs, paddings, mode='CONSTANT')
            return net

            
class SSD300(VGG16):
    def __init__(self):
        self.name = 'SSD300-VGG16'
        #self.base = 
        super(SSD300,self).__init__()
        self.match_threshold = 0.5
        self.negative_ratio  = 5
        
        
        
    def _ssd_multibox_layer(self,inputs,anchor_size,anchor_ratio):
        net          =  inputs
        num_anchors  =  len(anchor_ratio) *len(anchor_size)
        num_loc_pred =  num_anchors * 4
        loc_pred     =  slim.conv2d(net,num_loc_pred,[3,3], activation_fn = None,scope='conv_loc')
        loc_pred     =  tf.reshape(loc_pred,self.tensor_shape(loc_pred,rank=4)[:-1] + [num_anchors , 4] )
        
        num_cls_pred =  num_anchors * self.num_classes
        cls_pred     =  slim.conv2d(net,num_cls_pred,[3,3], activation_fn = None,scope='conv_cls')
        cls_pred     =  tf.reshape(cls_pred,self.tensor_shape(cls_pred,rank=4)[:-1] + [num_anchors , self.num_classes] )
        
        return cls_pred,loc_pred
    
    def _get_logits(self):
        self.predictions   = []
        self.logits        = []
        self.localizations = []
        for i , layer in enumerate(self.features):
            with tf.variable_scope(layer + '_box'):
                cp,lp =self._ssd_multibox_layer(self.end_points[layer],self.anchor_sizes[i],self.anchor_ratios[i])
                self.logits.append(cp)
                self.predictions.append(slim.softmax(cp))
                self.localizations.append(lp)
        
    
            
        
    def tensor_shape(self,x, rank=3):
        if x.get_shape().is_fully_defined():
            return x.get_shape().as_list()
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(x), rank)
            return [s if s is not None else d
                    for s, d in zip(static_shape, dynamic_shape)]    
    def _get_anchor_one_layer(self,feat_shape,sizes,ratios,step,offset = 0):
        dtype = np.float32
        img_shape = self.img_shape
    
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset) / feat_shape[0]
        x = (x.astype(dtype) + offset) / feat_shape[1]
        
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)
        num_anchors  =  (len(ratios)) + len(sizes)
        
        h = np.zeros((num_anchors, ), dtype=np.float32)
        w = np.zeros((num_anchors, ), dtype=np.float32)
        
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        h[1] = sizes[1] / img_shape[0]
        w[1] = sizes[1] / img_shape[1]
        #print(h)
        di = 2
        for i, r in enumerate(ratios):
            #print(i,r)
            #print(i+di)
            #h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            #w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
            h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
        di = i+di+1
        #for i, r in enumerate(ratios):
        #    
        #    h[i+di] = sizes[1] / img_shape[0] / math.sqrt(r)
        #    w[i+di] = sizes[1] / img_shape[1] * math.sqrt(r)
        return y, x, h, w
    def _get_anchor_all_layers(self):
        self.layers_anchors = []
        for i,s in enumerate(self.feat_shapes):
            anchor_box = self._get_anchor_one_layer(s,self.anchor_sizes[i], self.anchor_ratios[i],0,offset=0)
            self.layers_anchors.append(anchor_box)
            
    def _bboxes_encode_layer(self,labels,bboxes,anchor_layer,feat_shape,dtype=np.float32,offset=0):
        yref, xref, href, wref = anchor_layer
        if offset>0:
            yref = yref + offset/feat_shape[0]
            xref = xref + offset/feat_shape[1]
        ymin = yref - href / 2.
        xmin = xref - wref / 2.
        ymax = yref + href / 2.
        xmax = xref + wref / 2.
        anchor_volume = (xmax - xmin) * (ymax - ymin)
        
        shape = (yref.shape[0], yref.shape[1], href.size)
        feat_labels = np.zeros(shape, dtype=np.int64)
        feat_scores = np.zeros(shape, dtype=dtype)
        
        feat_ymin   = np.zeros(shape, dtype=dtype)
        feat_xmin   = np.zeros(shape, dtype=dtype)
        feat_ymax   = np.ones(shape, dtype=dtype)
        feat_xmax   = np.ones(shape, dtype=dtype)
        
        def jaccard(bbox):
            
            # (y,x) coordinate:
            #intersect_ymin = np.maximum(ymin,bbox[0])
            #intersect_xmin = np.maximum(xmin,bbox[1])
            #intersect_ymax = np.minimum(ymax,bbox[2])
            #intersect_xmax = np.minimum(xmax,bbox[3])
            
            intersect_xmin = np.maximum(xmin,bbox[0])
            intersect_ymin = np.maximum(ymin,bbox[1])
            intersect_xmax = np.minimum(xmax,bbox[2])
            intersect_ymax = np.minimum(ymax,bbox[3])
            
            intersect_height = np.maximum(intersect_ymax - intersect_ymin,0.)
            intersect_width  = np.maximum(intersect_xmax - intersect_xmin,0.)
            intersect_volume = intersect_height*intersect_width
            
            union_volume     = anchor_volume - intersect_volume + (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            jaccard          = intersect_volume/union_volume
            return jaccard
        def intersection_in_anchor(bbox):
            intersect_ymin = np.maximum(ymin,bbox[0])
            intersect_xmin = np.maximum(xmin,bbox[1])
            intersect_ymax = np.minimum(ymin,bbox[2])
            intersect_xmax = np.minimum(ymin,bbox[3])
            
            intersect_height = np.maximum(intersect_ymax - intersect_ymin,0.)
            intersect_width  = np.maximum(intersect_xmax - intersect_xmin,0.)
            intersect_volume = intersect_height*intersect_width
            
            score            = intersect_volume/anchor_volume
            return score
        for i in range(len(labels)):
            label  = labels[i]
            bbox   = bboxes[i]
            jcd    = jaccard(bbox)
            mask   = np.logical_and(jcd > 0.5 , jcd > feat_scores)
            mask_f = mask.astype(float)
            feat_scores = np.where(mask,jcd,feat_scores)
            feat_labels = mask_f*label     + (1-mask_f)*feat_labels # if mask is True, do first else do second
            
            # (y,x) coordinate:
            #feat_ymin   = mask_f*bbox[0]   + (1-mask_f)*feat_ymin
            #feat_xmin   = mask_f*bbox[1]   + (1-mask_f)*feat_xmin
            #feat_ymax   = mask_f*bbox[2]   + (1-mask_f)*feat_ymax
            #feat_xmax   = mask_f*bbox[3]   + (1-mask_f)*feat_xmax
            
            feat_xmin   = mask_f*bbox[0]   + (1-mask_f)*feat_xmin
            feat_ymin   = mask_f*bbox[1]   + (1-mask_f)*feat_ymin
            feat_xmax   = mask_f*bbox[2]   + (1-mask_f)*feat_xmax
            feat_ymax   = mask_f*bbox[3]   + (1-mask_f)*feat_ymax
            
        feat_cy    = (feat_ymin + feat_ymax)/2
        feat_cx    = (feat_xmin + feat_xmax)/2
        feat_height=  feat_ymax - feat_ymin
        feat_width =  feat_xmax - feat_xmin       
        #SSD Style
        feat_localization = [feat_cx,feat_cy,feat_width,feat_height]
        #My Style 
        #feat_localization = [feat_xmin,feat_ymin,feat_xmax,feat_ymax]
        return feat_labels,feat_localization,feat_scores
        
    def _bboxes_encode(self,labels,bboxes,offset=0):
        target_labels = []
        target_localizations = []
        target_scores = []
        for j in range(len(self.layers_anchors)):
            with tf.name_scope('bboxes_encode_block_{}'.format(j)):
                feat_labels,feat_localization,feat_scores   =self._bboxes_encode_layer(labels,bboxes,self.layers_anchors[j],self.feat_shapes[j],offset=0)
                for offset in [0.25,0.5,0.75]:
                    feat_labels2,feat_localization2,feat_scores2=self._bboxes_encode_layer(labels,bboxes,self.layers_anchors[j],self.feat_shapes[j],offset=offset)
                    
                    feat_mask2            = np.logical_and(feat_labels2>0,feat_scores<0.5)
                    feat_labels           = np.where(feat_mask2,feat_labels2         ,feat_labels)
                    feat_localization[0]  = np.where(feat_mask2,feat_localization2[0],feat_localization[0])
                    feat_localization[1]  = np.where(feat_mask2,feat_localization2[1],feat_localization[1])
                    feat_localization[2]  = np.where(feat_mask2,feat_localization2[2],feat_localization[2])
                    feat_localization[3]  = np.where(feat_mask2,feat_localization2[3],feat_localization[3])
                    feat_scores           = np.where(feat_mask2,feat_scores2         ,feat_scores)

                target_labels.append(feat_labels)
                target_localizations.append(feat_localization)
                target_scores.append(feat_scores)
        return target_labels,target_localizations,target_scores
    def _flatten_encoded(self,target_labels,target_localizations,target_scores):
        target_labels_flat = []
        for i in range(len(target_labels)):
            target_labels_flat.append(np.reshape(target_labels[i],[-1]))
        target_labels_flat=np.concatenate(target_labels_flat,axis=0)

        target_localizations_flat = []
        for i in range(len(target_localizations)):
            target_localizations_flat.append(np.reshape(target_localizations[i],[-1,4]))
        target_localizations_flat=np.concatenate(target_localizations_flat,axis=0)

        target_scores_flat = []
        for i in range(len(target_localizations)):
            target_scores_flat.append(np.reshape(target_scores[i],[-1]))
        target_scores_flat=np.concatenate(target_scores_flat,axis=0)
        
        return target_labels_flat,target_localizations_flat,target_scores_flat
    def _make_batch(self,batch_labels,batch_bboxes):
        batch_size = self.batch_size
        batch_labels_flat = []
        batch_localizations_flat = []
        batch_scores_flat = []
        
        for b in range(len(batch_labels)):
            labels=batch_labels[b]
            bboxes=batch_bboxes[b]
            target_labels,target_localizations,target_scores = self._bboxes_encode(labels,bboxes)
            target_labels_flat,target_localizations_flat,target_scores_flat = self._flatten_encoded(target_labels,target_localizations,target_scores)
            
            batch_labels_flat.append(target_labels_flat)
            batch_localizations_flat.append(target_localizations_flat)
            batch_scores_flat.append(target_scores_flat)
            
        batch_labels_flat = np.array(batch_labels_flat)
        batch_localizations_flat = np.array(batch_localizations_flat)
        batch_scores_flat = np.array(batch_scores_flat)
        
        return batch_labels_flat,batch_localizations_flat,batch_scores_flat
           
        
            
    def _get_losses(self,gt_localization,gt_classes,gt_scores,alpha=1.):
        num_cls = self.num_classes
        logits_flat = []
        localizations_flat = []
        for t in range(len(self.logits)):
            logits_flat.append(tf.reshape(self.logits[t],[-1,8732,num_cls]))
            localizations_flat.append(tf.reshape(self.localizations[t],[-1,8732,4]))

        logits_flat        = tf.concat(logits_flat, axis=0)
        localizations_flat = tf.concat(localizations_flat, axis=0)
        logits             = logits_flat
        with tf.name_scope('ssd_losses'):
            
            pos_mask    = gt_scores > self.match_threshold
            pos_mask_f  = tf.cast(pos_mask,tf.float32)
            bg_classes  = tf.cast(pos_mask,tf.int32)
            num_positive= tf.reduce_sum(pos_mask_f)
            
            neg_mask    =  tf.logical_not(pos_mask)
            neg_mask_f  =  tf.cast(neg_mask,tf.float32)
            num_negative=  num_positive * self.negative_ratio +self.batch_size
            num_negative=  tf.cast(num_negative,tf.int32)
            predictions =  slim.softmax(logits)
            neg_score   =  tf.where(neg_mask,predictions[:,:,0],1-neg_mask_f)
            #neg_score_flat=tf.reshape(neg_score,[-1])
            val,idxs      =tf.nn.top_k(-neg_score,k=num_negative)
            max_hard_pred = -val[-1]
            neg_mask      = tf.logical_and(neg_mask,neg_score<max_hard_pred)
            
            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=gt_classes)
                loss = tf.reduce_sum(loss*pos_mask_f)/self.batch_size
                self.pos_conf_los = loss
                tf.losses.add_loss(loss)
            with tf.name_scope('cross_entropy_neg'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=bg_classes)
                loss = tf.reduce_sum(loss*neg_mask_f)/self.batch_size
                self.neg_conf_loss = loss
                tf.losses.add_loss(loss)  
            with tf.name_scope('localization'):
                pos_mask_f_loc = tf.expand_dims(pos_mask_f,axis=-1)
                loss = self.smooth_L1(localizations_flat - gt_localization)
                loss = tf.reduce_sum(loss*pos_mask_f_loc)/self.batch_size
                self.loc_loss = loss
                tf.losses.add_loss(loss)
            self.total_loss = self.pos_conf_los+self.neg_conf_loss+self.loc_loss
    def _get_placeholders(self):
        self.Y_localizations = []
        self.Y_classes       = []
        num_cls = self.num_classes
        #self.y_scores        = []

        Y_class         = tf.placeholder(tf.float32,[None,38,38,4,num_cls],name='Y_class_block04')
        Y_localization  = tf.placeholder(tf.float32,[None,38,38,4,4]      ,name='Y_local_block04')
        self.Y_classes.append(Y_class)
        self.Y_localizations.append(Y_localization)

        Y_class         = tf.placeholder(tf.float32,[None,19,19,8,num_cls],name='Y_class_block07')
        Y_localization  = tf.placeholder(tf.float32,[None,19,19,8,4]      ,name='Y_local_block07')
        self.Y_classes.append(Y_class)
        self.Y_localizations.append(Y_localization)

        Y_class         = tf.placeholder(tf.float32,[None,10,10,8,num_cls],name='Y_class_block08')
        Y_localization  = tf.placeholder(tf.float32,[None,10,10,8,4]      ,name='Y_local_block08')
        self.Y_classes.append(Y_class)
        self.Y_localizations.append(Y_localization)

        Y_class         = tf.placeholder(tf.float32,[None,5,5,8,num_cls],name='Y_class_block09')
        Y_localization  = tf.placeholder(tf.float32,[None,5,5,8,4]      ,name='Y_local_block09')
        self.Y_classes.append(Y_class)
        self.Y_localizations.append(Y_localization)

        Y_class         = tf.placeholder(tf.float32,[None,3,3,4,num_cls],name='Y_class_block10')
        Y_localization  = tf.placeholder(tf.float32,[None,3,3,4,4]      ,name='Y_local_block10')
        self.Y_classes.append(Y_class)
        self.Y_localizations.append(Y_localization)

        Y_class         = tf.placeholder(tf.float32,[None,2,2,4,num_cls],name='Y_class_block11')
        Y_localization  = tf.placeholder(tf.float32,[None,2,2,4,4]      ,name='Y_local_block11')
        self.Y_classes.append(Y_class)
        self.Y_localizations.append(Y_localization)
        
        
    def _generate_gt(self,sess,Y_cls,Y_loc,Y_score):
        self.gt_localization = sess.run()
                
    def smooth_L1(self,x):  
        absx  = tf.abs(x)
        minx=tf.minimum(absx,1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r 
    
            
            
            
            
            

                
        
        
            