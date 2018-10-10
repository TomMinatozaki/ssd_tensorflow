import tensorflow as tf
import numpy as np
import os 
import math
import cv2
slim = tf.contrib.slim
from VGG import *

class SSD300(VGG16):
    def __init__(self,with_batch_norm=True):
        self.name = 'SSD300-VGG16'
        
        
        #self.base = 
        self.TRAINING  = tf.placeholder(tf.bool)
        super(SSD300,self).__init__(is_training=self.TRAINING,with_batch_norm=with_batch_norm)
        self.match_threshold = 0.4
        self.jaccard_threshold = 0.4
        self.bg_threshold    = 0.25
        self.negative_ratio  = 5
        self.train_data_path = r'D:\Data\VOC\VOC2007\train\JPEGImages'
        self.test_data_path  = r'D:\Data\VOC\VOC2007\test\JPEGImages'
        self.pretrained_path = r'D:\Data\pretrained\vgg_16.ckpt'
        #self.num_anchors     =  len(anchor_ratio) + 2*len(anchor_size)
        self.flatten_len_list  = [5776,2166,600,150,36,16]
        self.flatten_len_list  = [self.feat_shapes[t][0]*self.feat_shapes[t][1]*(len(self.anchor_sizes[t])+2*len(self.anchor_ratios[t])) for t in range(6)]
        self.flatten_num = sum(self.flatten_len_list)
        #flatten_num = 8744
        self.Y_cls     = tf.placeholder(tf.int64,[None,self.flatten_num])
        self.Y_loc     = tf.placeholder(tf.float32,[None,self.flatten_num,4])
        self.Y_score   = tf.placeholder(tf.float32,[None,self.flatten_num])
        self.global_step =  tf.Variable(0,trainable=False,name='global_step')
        self.transform = Transform()
        
        
    def _ssd_multibox_layer(self,inputs,anchor_size,anchor_ratio):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding='SAME'):
            with slim.arg_scope([slim.conv2d,slim.fully_connected],activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(0.0001)):
                net          =  inputs
                num_anchors  =  2*len(anchor_ratio) + len(anchor_size)
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
                
                
        num_cls = self.num_classes
        self.logits_flat = []
        self.localizations_flat = []
        self.predictions_flat = []
        with tf.name_scope('ssd_reshapes_flattened'):
            for t in range(len(self.logits)):
                self.logits_flat.append(tf.reshape(self.logits[t],[-1,tf.reduce_prod(self.logits[t].shape[1:4]),num_cls],name='reshape_logit'+str(t)))
                self.localizations_flat.append(tf.reshape(self.localizations[t],[-1,tf.reduce_prod(self.localizations[t].shape[1:4]),4],name='reshape_loc'+str(t)))
                self.predictions_flat.append(tf.reshape(self.predictions[t],[-1,tf.reduce_prod(self.predictions[t].shape[1:4]),num_cls],name='reshape_pred'+str(t)))
            self.logits_flat        = tf.concat(self.logits_flat       , axis=1)
            self.localizations_flat = tf.concat(self.localizations_flat, axis=1)
            self.predictions_flat   = tf.concat(self.predictions_flat  , axis=1)
            
        
    def tensor_shape(self,x, rank=3):
        if x.get_shape().is_fully_defined():
            return x.get_shape().as_list()
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(x), rank)
            return [s if s is not None else d
                    for s, d in zip(static_shape, dynamic_shape)]    
    def _get_anchor_one_layer(self,feat_shape,sizes,ratios,step,offset = [0,0]):
        dtype = np.float32
        img_shape = self.img_shape
    
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset[0]) / feat_shape[0]
        x = (x.astype(dtype) + offset[1]) / feat_shape[1]
        
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)
        num_anchors  =  2*(len(ratios))  + len(sizes) 
        
        h = np.zeros((num_anchors, ), dtype=np.float32)
        w = np.zeros((num_anchors, ), dtype=np.float32)
        
        h[0] = sizes[0] / img_shape[0]
        w[0] = sizes[0] / img_shape[1]
        h[1] = sizes[1] / img_shape[0]
        w[1] = sizes[1] / img_shape[1]
        di = 2
        if len(sizes)>2:
            h[2] = sizes[2] / img_shape[0]
            w[2] = sizes[2] / img_shape[1]
            di = 3
        #print(h)
        
        for i, r in enumerate(ratios):
            #print(i,r)
            #print(i+di)
            #h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            #w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
            h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
        di = i+di+1
        
        for i, r in enumerate(ratios):
            #print(i,r)
            #print(i+di)
            #h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
            #w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
            h[i+di] = sizes[1] / img_shape[0] / math.sqrt(r)
            w[i+di] = sizes[1] / img_shape[1] * math.sqrt(r)
        di = i+di+1
        
        #h  = np.where(x+h>0,np.ones_like(x)-x,h)
        #w  = np.where(y+w>0,np.ones_like(y)-y,w)
        #for i, r in enumerate(ratios):
        #    
        #    h[i+di] = sizes[1] / img_shape[0] / math.sqrt(r)
        #    w[i+di] = sizes[1] / img_shape[1] * math.sqrt(r)
        return y, x, h, w
    def _get_anchor_all_layers(self):
        self.layers_anchors = []
        for i,s in enumerate(self.feat_shapes):
            anchor_box = self._get_anchor_one_layer(s,self.anchor_sizes[i], self.anchor_ratios[i],0,offset=[0,0])
            self.layers_anchors.append(anchor_box)
            
    def _bboxes_encode_layer(self,labels,bboxes,anchor_layer,feat_shape,dtype=np.float32,offset=[0,0]):
        yref, xref, href, wref = anchor_layer
        
        yref_flat = np.reshape(yref,([-1]))
        xref_flat = np.reshape(xref,([-1]))
        href_flat = np.reshape(href,([-1]))
        wref_flat = np.reshape(wref,([-1]))
      
        href_flat = np.repeat(href_flat,len(yref_flat))
        wref_flat = np.repeat(wref_flat,len(xref_flat))
        xref_flat = np.tile(xref_flat,len(wref))
        yref_flat = np.tile(yref_flat,len(href))
    
        yref_off = yref_flat + offset[0]/feat_shape[0]
        xref_off = xref_flat + offset[1]/feat_shape[1]
        ymin = yref_off - href_flat / 2.
        xmin = xref_off - wref_flat / 2.
        ymax = yref_off + href_flat / 2.
        xmax = xref_off + wref_flat / 2.
        
        xmin  = np.where(xmin <0 ,np.zeros_like(xmin),xmin)
        xmax  = np.where(xmax >1 ,np.ones_like(xmax) ,xmax)
        ymin  = np.where(ymin <0 ,np.zeros_like(ymin),ymin)
        ymax  = np.where(ymax >1 ,np.ones_like(ymax) ,ymax)

        anchor_volume = (xmax - xmin) * (ymax - ymin)
        
        shape = (yref.shape[0], yref.shape[1], href.size)
        feat_labels = np.zeros_like(ymin, dtype=dtype)
        feat_scores = np.zeros_like(ymin, dtype=dtype)
        
        feat_ymin   = np.zeros_like(ymin, dtype=dtype)
        feat_xmin   = np.zeros_like(xmin, dtype=dtype)
        feat_ymax   = np.zeros_like(ymax, dtype=dtype)
        feat_xmax   = np.zeros_like(xmax, dtype=dtype)
        
        def jaccard(bbox):
            
            # (y,x) coordinate:
           
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
            intersect_ymin = np.maximum(ymin,bbox[1])
            intersect_xmin = np.maximum(xmin,bbox[0])
            intersect_ymax = np.minimum(ymax,bbox[3])
            intersect_xmax = np.minimum(xmax,bbox[2])
            
            intersect_height = np.maximum(intersect_ymax - intersect_ymin,0.)
            intersect_width  = np.maximum(intersect_xmax - intersect_xmin,0.)
            intersect_volume = intersect_height*intersect_width
            
            score            = intersect_volume/anchor_volume
            return score
        for i in range(len(labels)):
            label  = labels[i]
            bbox   = bboxes[i]
            jcd    = jaccard(bbox)
            mask   = np.logical_and(jcd > self.jaccard_threshold , jcd > feat_scores)
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
       
        
        feat_xmin  = feat_xmin.astype(np.float32)
        feat_xmax  = feat_xmax.astype(np.float32)
        feat_ymin  = feat_ymin.astype(np.float32)
        feat_ymax  = feat_ymax.astype(np.float32)
        
        
        feat_cy    = np.where(feat_ymin>0,((feat_ymax + feat_ymin)/2 -yref_flat),np.zeros_like(feat_ymin))  / href_flat
        feat_cx    = np.where(feat_xmin>0,((feat_xmax + feat_xmin)/2 -xref_flat),np.zeros_like(feat_xmin))  / wref_flat
        feat_height= np.where(feat_ymin>0, np.log( (feat_ymax - feat_ymin) / href_flat) ,np.zeros_like(feat_ymin))
        feat_width = np.where(feat_xmin>0, np.log( (feat_xmax - feat_xmin) / wref_flat) ,np.zeros_like(feat_xmin))
        #SSD Style
       
        feat_localization = [feat_cx,feat_cy,feat_width,feat_height]
        #My Style 
        #feat_localization = [feat_xmin,feat_ymin,feat_xmax,feat_ymax]
        return feat_labels,feat_localization,feat_scores
        
    def _bboxes_encode(self,labels,bboxes,offsets=[0.25,0.5,0.75]):
        
        target_labels = []
        target_localizations = []
        target_scores = []
        target_xmin = []
        target_ymin = []
        target_xmax = []
        target_ymax = []
        bboxes = np.array(bboxes)
        bboxes = bboxes/300
        for j in range(len(self.layers_anchors)):
            with tf.name_scope('bboxes_encode_block_{}'.format(j)):
                feat_labels,feat_localization,feat_scores   =self._bboxes_encode_layer(labels,bboxes,self.layers_anchors[j],self.feat_shapes[j])
                for offset_x in offsets:
                    for offset_y in offsets:
                     
                        offset = [offset_x,offset_y]
                        feat_labels2,feat_localization2,feat_scores2=self._bboxes_encode_layer(labels,bboxes,self.layers_anchors[j],self.feat_shapes[j],offset=offset)
                        
                        feat_mask2            = np.logical_and(feat_labels2>0,feat_labels<0.5)
                        feat_mask2            = np.logical_or(feat_mask2,feat_scores2>feat_scores)

                        feat_labels           = np.where(feat_mask2,feat_labels2         ,feat_labels)
                        feat_localization[0]  = np.where(feat_mask2,feat_localization2[0],feat_localization[0])
                        feat_localization[1]  = np.where(feat_mask2,feat_localization2[1],feat_localization[1])
                        feat_localization[2]  = np.where(feat_mask2,feat_localization2[2],feat_localization[2])
                        feat_localization[3]  = np.where(feat_mask2,feat_localization2[3],feat_localization[3])
                        feat_scores           = np.where(feat_mask2,feat_scores2         ,feat_scores)
                    
                target_labels.append(feat_labels)
                target_xmin.append(feat_localization[0])
                target_ymin.append(feat_localization[1])
                target_xmax.append(feat_localization[2])
                target_ymax.append(feat_localization[3])
                #target_localizations.append(feat_localization)
                target_scores.append(feat_scores)
        target_labels        =  np.concatenate(target_labels)
        target_xmin          =  np.concatenate(target_xmin)
        target_ymin          =  np.concatenate(target_ymin)
        target_xmax          =  np.concatenate(target_xmax)
        target_ymax          =  np.concatenate(target_ymax)
        #target_localizations =  np.concatenate(target_localizations,axis=1)
        target_scores        =  np.concatenate(target_scores)
        target_localizations =  np.zeros(list(np.shape(target_xmin))+[4])
        target_localizations[:,0] = target_xmin
        target_localizations[:,1] = target_ymin
        target_localizations[:,2] = target_xmax
        target_localizations[:,3] = target_ymax
        return target_labels,target_localizations,target_scores
    def _flatten_encoded(self,target_labels,target_localizations,target_scores):
        target_labels_flat = []
        for i in range(len(target_labels)):
            target_labels_flat.append(np.reshape(target_labels[i],[-1]))
        
        target_labels_flat=np.concatenate(target_labels_flat,axis=0)

        target_localizations_flat = []
        for i in range(len(target_localizations)):
            target_localizations_flat.append(np.reshape(target_localizations[i],[-1,4],order='F'))
            
        target_localizations_flat=np.concatenate(target_localizations_flat,axis=0)

        target_scores_flat = []
        for i in range(len(target_localizations)):
            target_scores_flat.append(np.reshape(target_scores[i],[-1]))
        target_scores_flat=np.concatenate(target_scores_flat,axis=0)
        
        return target_labels_flat,target_localizations_flat,target_scores_flat
    def _make_batch(self):
        batch_size = self.batch_size
        batch_labels_flat = []
        batch_localizations_flat = []
        batch_scores_flat = []
        
        batch_labels = self.batch_labels
        batch_bboxes = self.batch_bboxes
      
        for b in range(len(batch_labels)):
            labels=batch_labels[b]
            bboxes=batch_bboxes[b]
            target_labels_flat,target_localizations_flat,target_scores_flat = self._bboxes_encode(labels,bboxes)
            #target_labels_flat,target_localizations_flat,target_scores_flat = self._flatten_encoded(target_labels,target_localizations,target_scores)
            #print(target_localizations_flat.shape)
            
            batch_labels_flat.append(target_labels_flat)
            batch_localizations_flat.append(target_localizations_flat)
            batch_scores_flat.append(target_scores_flat)
            
        batch_labels_flat        = np.array(batch_labels_flat)
        batch_localizations_flat = np.array(batch_localizations_flat)
        batch_scores_flat        = np.array(batch_scores_flat)
        
        #b=0
        #labels=batch_labels[b]
        #bboxes=batch_bboxes[b]
        #target_labels,target_localizations,target_scores = self._bboxes_encode(labels,bboxes)
        
        #self.local_temp = target_localizations
        
        return batch_labels_flat,batch_localizations_flat,batch_scores_flat
    def _import_data(self,data_raw,key_list,is_test=False):
            batch_bboxes = []
            batch_labels = []
            batch_imgs   = []
            data_path = self.train_data_path
            
            if is_test:
                data_path = self.test_data_path
            for k in range(len(key_list)):
                key    = key_list[k]
                bboxes = data_raw[key]['box_coords']
                labels = data_raw[key]['class']
                
                img    = cv2.imread(os.path.join(data_path,key))
                img    = cv2.resize(img,(300,300))
                img    = img/255
                #x_ratio = 300./img.shape[0]
                #y_ratio = 300./img.shape[1]
                bboxes = np.array(bboxes)
                batch_bboxes.append(bboxes)
                batch_labels.append(labels)
                batch_imgs.append(img)
            self.batch_bboxes = batch_bboxes
            self.batch_labels = batch_labels
            self.batch_imgs   = np.array(batch_imgs)
            self.batch_keys   = key_list
    def _train_one_step(self,sess,data_raw,key_list,does_print_loss=False,do_flip=None,do_transform = False):
        self._import_data(data_raw,key_list)
        if do_transform:
            self.transform._new_batch(self.batch_imgs,self.batch_bboxes)
            self.transform._brightness()
            self.transform._contrast()
            self.transform._hue()
            self.transform._saturation()
            self.transform._flip()
            dice =np.random.random()
            if dice < 0.25:
                self.transform._expand()
            if dice > 0.75:
                self.transform._randcrop()
            self.batch_imgs   = self.transform.imgs
            self.batch_bboxes = self.transform.box_case
        batch_labels_flat,batch_localizations_flat,batch_scores_flat=self._make_batch()
        feed_dict = {self.X:self.batch_imgs,self.Y_cls:batch_labels_flat,self.Y_loc:batch_localizations_flat,self.Y_score:batch_scores_flat,self.TRAINING:True}
        sess.run(self.optimizer,feed_dict=feed_dict)
        if does_print_loss:
            loss_now_total = sess.run(self.total_loss,feed_dict=self.feed_dict_fix)
            loss_now_loc   = sess.run(self.loc_loss  ,feed_dict=self.feed_dict_fix)
            loss_now_pos   = sess.run(self.pos_conf_los  ,feed_dict=self.feed_dict_fix)
            loss_now_reg   = sess.run(self.reg_loss  ,feed_dict=self.feed_dict_fix)
            print('total loss: {} '.format(loss_now_total))
            print('loc   loss: {} '.format(loss_now_loc))
            print('pos   loss: {} '.format(loss_now_pos))
            print('reg   loss: {} '.format(loss_now_reg))
    def _get_fixed_train_data(self,data_raw,key_list):
        self._import_data(data_raw,key_list)
        batch_labels_flat,batch_localizations_flat,batch_scores_flat=self._make_batch()
        self.feed_dict_fix = {self.X:self.batch_imgs,self.Y_cls:batch_labels_flat,self.Y_loc:batch_localizations_flat,self.Y_score:batch_scores_flat,self.TRAINING:False}        
    def _get_test_report(self,sess,data_raw_test,key_list_test):
        self._import_data(data_raw_test,key_list_test)
        batch_labels_flat,batch_localizations_flat,batch_scores_flat=self._make_batch()
        feed_dict = {self.X:self.batch_imgs,self.Y_cls:batch_labels_flat,self.Y_loc:batch_localizations_flat,self.Y_score:batch_scores_flat,self.TRAINING:False}
        loss_now_total = sess.run(self.total_loss,feed_dict=feed_dict)
        loss_now_loc   = sess.run(self.loc_loss  ,feed_dict=feed_dict)
        print('total loss: {} '.format(loss_now_total))
        print('loc   loss: {} '.format(loss_now_loc))
    def _get_losses(self,alpha=10.):
    
        gt_localization = self.Y_loc
        gt_classes      = self.Y_cls
        gt_scores       = self.Y_score
        num_cls = self.num_classes
        
        logits             = self.logits_flat
        localizations_flat = self.localizations_flat
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
            val,idxs      = tf.nn.top_k(-neg_score,k=num_negative)
            max_hard_pred = -val[:,-1]
            max_hard_pred = tf.expand_dims(max_hard_pred,axis=-1)
            max_hard_pred = tf.tile(max_hard_pred,[1,self.flatten_num])
            neg_mask      = tf.logical_and(neg_mask,neg_score < max_hard_pred)
            neg_mask_f    =  tf.cast(neg_mask,tf.float32)
            
            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=gt_classes,name='pos_loss')
                self.pos_conf_los = tf.reduce_sum(loss*pos_mask_f)/self.batch_size
                
                tf.losses.add_loss(self.pos_conf_los)
            with tf.name_scope('cross_entropy_neg'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=bg_classes,name='neg_loss')
                self.neg_conf_loss = tf.reduce_sum(loss*neg_mask_f)/self.batch_size
                
                tf.losses.add_loss(self.neg_conf_loss)  
            with tf.name_scope('localization'):
                pos_mask_f_loc = tf.expand_dims(pos_mask_f,axis=-1)
                loss = self.smooth_L1(localizations_flat - gt_localization)
                loss = tf.reduce_sum(loss*pos_mask_f_loc)/self.batch_size
                self.loc_loss = loss*alpha
                tf.losses.add_loss(loss)
            self.reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
            self.pos_mask = pos_mask
            self.neg_mask = neg_mask
            self.total_loss = self.pos_conf_los+self.neg_conf_loss+self.loc_loss+self.reg_loss
    def _get_optimizer(self,learning_rate=0.0001):
        self.lr_decay = tf.train.exponential_decay(learning_rate,self.global_step,50000,0.96,staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_decay,name='adam_decay').minimize(self.total_loss,global_step=self.global_step)
    def _decode_from_logits(self,sess,data_raw,key_list,is_test = False):
        self._import_data(data_raw,key_list,is_test = is_test)
        batch_labels_flat,batch_localizations_flat,batch_scores_flat=self._make_batch()
        feed_dict = {self.X:self.batch_imgs,self.Y_cls:batch_labels_flat,self.Y_loc:batch_localizations_flat,self.Y_score:batch_scores_flat,self.TRAINING:False}
        loc = sess.run(self.localizations_flat,feed_dict=feed_dict)
        pre = sess.run(self.predictions_flat  ,feed_dict=feed_dict)
        t = 1
        cls_preds = []
        cut_idx = 0
        batch_bboxes_test= []
        batch_bboxes_cand = []
        batch_cls = []
        batch_cls_score = []
        for b in range(len(key_list)):
            #print(b)
            bboxes_test=[]
            clses = []
            cls_scores = []
            cut_idx = 0
            for k in range(5):
                #print(k)
                feat_localization = loc[b,cut_idx:cut_idx+self.flatten_len_list[k],:]
                feat_pre          = pre[b,cut_idx:cut_idx+self.flatten_len_list[k],:]
                feat_bg_score     = feat_pre[:,0]
                cut_idx = cut_idx+self.flatten_len_list[k]
                #print(cut_idx)
                #print(feat_localization.shape)
                
                yref, xref, href, wref = self.layers_anchors[k]

                yref_flat = np.reshape(yref,([-1]))
                xref_flat = np.reshape(xref,([-1]))
                href_flat = np.reshape(href,([-1]))
                wref_flat = np.reshape(wref,([-1]))
                
               
                
                href_flat = np.repeat(href_flat,len(yref_flat))
                wref_flat = np.repeat(wref_flat,len(xref_flat))
                xref_flat = np.tile(xref_flat,len(wref))
                yref_flat = np.tile(yref_flat,len(href))
            

                
                feat_not_bg=feat_bg_score<self.bg_threshold
                feat_cx = np.where(feat_not_bg,feat_localization[:,0],np.zeros_like(feat_localization[:,0]))
                feat_cy = np.where(feat_not_bg,feat_localization[:,1],np.zeros_like(feat_localization[:,0]))
                feat_w  = np.where(feat_not_bg,feat_localization[:,2],np.zeros_like(feat_localization[:,0]))
                feat_h  = np.where(feat_not_bg,feat_localization[:,3],np.zeros_like(feat_localization[:,0]))
                
                feat_cx = np.where(feat_not_bg,feat_cx*wref_flat+xref_flat,np.zeros_like(feat_cx))#
                feat_cy = np.where(feat_not_bg,feat_cy*href_flat+yref_flat,np.zeros_like(feat_cy))#
                #feat_cy = feat_cy*href_flat+yref_flat
                feat_w  = np.where(feat_not_bg , np.exp(feat_w)*wref_flat,np.zeros_like(feat_w))#*wref_flat
                feat_h  = np.where(feat_not_bg , np.exp(feat_h)*href_flat,np.zeros_like(feat_h)) #*href_flat
                
                
                argwhere = np.logical_or(feat_w>0.001 , feat_w<-0.001)
                bboxes_center=[feat_cx[argwhere],feat_cy[argwhere]  ,feat_w[argwhere]  ,feat_h[argwhere]]
                cls       = np.argmax(feat_pre[argwhere],axis=1)
                cls_score = np.max(feat_pre[argwhere],axis=1)
                clses.append(cls)
                cls_scores.append(cls_score)
                #feat_not_bg_expand = np.expand_dims(feat_not_bg,axis=-1)
                #feat_not_bg_expand = np.tile(feat_not_bg_expand,24)
                #feat_logits_notbg=np.where(feat_not_bg_expand,feat_logits,np.zeros_like(feat_logits))
                #feat_cls = np.argmax(feat_logits_notbg,axis=-1)
                #cls_pred = feat_cls[np.nonzero(feat_cls)]
                #if len(cls_pred)>0:
                #    cls_preds.append(cls_pred)
                for i in range(len(bboxes_center[3])):
                    bbox = np.zeros(4)
                    bbox[0] = bboxes_center[0][i]-bboxes_center[2][i]/2
                    bbox[1] = bboxes_center[1][i]-bboxes_center[3][i]/2
                    bbox[2] = bboxes_center[0][i]+bboxes_center[2][i]/2
                    bbox[3] = bboxes_center[1][i]+bboxes_center[3][i]/2
                    bboxes_test.append(bbox)
                    #print(k)
            batch_bboxes_test.append(bboxes_test)
            batch_bboxes_cand.append(bboxes_center)
            batch_cls.append(np.concatenate(clses))
            batch_cls_score.append(np.concatenate(cls_scores))
        return batch_bboxes_test,batch_bboxes_cand,batch_cls,batch_cls_score
        pass
    def _decode_from_batch(self,data_raw,key_list,is_test=False):
        self._import_data(data_raw,key_list,is_test=is_test)
        batch_labels_flat,batch_localizations_flat,batch_scores_flat=self._make_batch()
        
        b = 1
        
        cls_preds = []
        cut_idx = 0
        batch_bboxes_test= []
        batch_bboxes_cand = []
        batch_cls = []
        for b in range(len(key_list)):
            #print(b)
            bboxes_test=[]
            clses = []
            cut_idx = 0
            for k in range(5):
                #print(k)
                feat_localization = batch_localizations_flat[b,cut_idx:cut_idx+self.flatten_len_list[k],:]
                feat_scores       = batch_scores_flat[b,cut_idx:cut_idx+self.flatten_len_list[k]]
                feat_labels       = batch_labels_flat[b,cut_idx:cut_idx+self.flatten_len_list[k]]
                cut_idx = cut_idx+self.flatten_len_list[k]
                #print(cut_idx)
                #print(feat_localization.shape)
                
                yref, xref, href, wref = self.layers_anchors[k]

                yref_flat = np.reshape(yref,([-1]))
                xref_flat = np.reshape(xref,([-1]))
                href_flat = np.reshape(href,([-1]))
                wref_flat = np.reshape(wref,([-1]))
                
               
                
                href_flat = np.repeat(href_flat,len(yref_flat))
                wref_flat = np.repeat(wref_flat,len(xref_flat))
                xref_flat = np.tile(xref_flat,len(wref))
                yref_flat = np.tile(yref_flat,len(href))
            

                
                feat_not_bg=feat_scores>0.5
                feat_cx = np.where(feat_not_bg,feat_localization[:,0],np.zeros_like(feat_localization[:,0]))
                feat_cy = np.where(feat_not_bg,feat_localization[:,1],np.zeros_like(feat_localization[:,0]))
                feat_w  = np.where(feat_not_bg,feat_localization[:,2],np.zeros_like(feat_localization[:,0]))
                feat_h  = np.where(feat_not_bg,feat_localization[:,3],np.zeros_like(feat_localization[:,0]))
                
                #feat_cx = feat_cx+xref_flat
                #feat_cy = feat_cy+yref_flat
                #feat_w  = feat_w #*wref_flat
                #feat_h  = feat_h #*href_flat
                feat_cx = feat_cx*wref_flat+xref_flat
                feat_cy = feat_cy*href_flat+yref_flat
                feat_w  = np.where(feat_not_bg , np.exp(feat_w)*wref_flat,np.zeros_like(feat_w))#*wref_flat
                feat_h  = np.where(feat_not_bg , np.exp(feat_h)*href_flat,np.zeros_like(feat_h)) #*href_flat
                
                
                argwhere = np.logical_or(feat_w>0.00001 , feat_w<-0.00001)
                bboxes_center=[feat_cx[argwhere],feat_cy[argwhere]  ,feat_w[argwhere]  ,feat_h[argwhere]]
                cls = feat_labels[argwhere]
                clses.append(cls)
                #feat_not_bg_expand = np.expand_dims(feat_not_bg,axis=-1)
                #feat_not_bg_expand = np.tile(feat_not_bg_expand,24)
                #feat_logits_notbg=np.where(feat_not_bg_expand,feat_logits,np.zeros_like(feat_logits))
                #feat_cls = np.argmax(feat_logits_notbg,axis=-1)
                #cls_pred = feat_cls[np.nonzero(feat_cls)]
                #if len(cls_pred)>0:
                #    cls_preds.append(cls_pred)
                for i in range(len(bboxes_center[3])):
                    bbox = np.zeros(4)
                    bbox[0] = bboxes_center[0][i]-bboxes_center[2][i]/2
                    bbox[1] = bboxes_center[1][i]-bboxes_center[3][i]/2
                    bbox[2] = bboxes_center[0][i]+bboxes_center[2][i]/2
                    bbox[3] = bboxes_center[1][i]+bboxes_center[3][i]/2
                    bboxes_test.append(bbox)
                    #print(k)
            
            
            batch_cls.append(np.concatenate(clses))
            batch_bboxes_test.append(bboxes_test)
            batch_bboxes_cand.append(bboxes_center)
        return batch_bboxes_test,batch_bboxes_cand,batch_cls
        pass
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
    def  nms(self,batch_bboxes,batch_cls,batch_cls_score):
        def jaccard(box1,box2):
            xmin1 = box1[0]
            ymin1 = box1[1]
            xmax1 = box1[2]
            ymax1 = box1[3]
            
            xmin2 = box2[0]
            ymin2 = box2[1]
            xmax2 = box2[2]
            ymax2 = box2[3]
            
            xmin_inter = np.maximum(xmin1,xmin2)
            ymin_inter = np.maximum(ymin1,ymin2)
            xmax_inter = np.minimum(xmax1,xmax2)
            ymax_inter = np.minimum(ymax1,ymax2)
            
            
            if (xmax_inter-xmin_inter)<0  or (ymax_inter-ymin_inter) < 0:
               return 0
               
            inter_volume = (xmax_inter-xmin_inter)*(ymax_inter-ymin_inter)
            union_volume = (xmax1-xmin1)*(ymax1-ymin1)+(xmax2-xmin2)*(ymax2-ymin2)-inter_volume
            jaccard = inter_volume/union_volume
            return jaccard    
        batch_nms_bboxes= []
        batch_nms_cls   = []
        batch_nms_score  = []
        #for b in range(self.batch_size):
        for b in range(len(batch_bboxes)):
            
            remained_boxes = np.copy(batch_bboxes[b])
            remained_score = np.copy(batch_cls_score[b])
            remained_cls   = np.copy(batch_cls[b])
            nmsed_boxes    = []
            nmsed_cls      = []
            nmsed_score    = []
            
            if len(remained_boxes) == 0:
                batch_nms_bboxes.append(nmsed_boxes)
                batch_nms_cls.append(nmsed_cls)
                batch_nms_score.append(nmsed_score)
                continue

            
            for _ in range(1000):
                highest_idx    = np.argmax(remained_score)
                highest_box    = remained_boxes[highest_idx]
                highest_score  = remained_score[highest_idx]
                highest_cls    = remained_cls[highest_idx]
                
                if highest_cls == 0:
                    remained_boxes=np.delete(remained_boxes,[highest_idx],axis=0)
                    remained_score=np.delete(remained_score,[highest_idx],axis=0)
                    remained_cls  =np.delete(remained_cls  ,[highest_idx],axis=0)
                    if len(remained_boxes) == 0:
                        break
                    continue
                    
                out_idx        = []
                for k in range(len(remained_boxes)):
                    jac = jaccard(remained_boxes[k],highest_box)
                    if jac > 0.5 and remained_cls[k] == highest_cls:
                        out_idx.append(k)

                remained_boxes=np.delete(remained_boxes,out_idx,axis=0)
                remained_score=np.delete(remained_score,out_idx,axis=0)
                remained_cls  =np.delete(remained_cls,out_idx,axis=0)
                
                nmsed_boxes.append(highest_box)
                nmsed_cls.append(highest_cls)
                nmsed_score.append(highest_score)
                if len(remained_boxes) == 0:
                    break
          
            batch_nms_bboxes.append(nmsed_boxes)
            batch_nms_cls.append(nmsed_cls)
            batch_nms_score.append(nmsed_score)
        self.batch_nms_bboxes = batch_nms_bboxes
        self.batch_nms_cls    = batch_nms_cls
        self.batch_nms_score  = batch_nms_score
        
    def _get_accuracy(self,sess,data_raw,key_list,is_test=False):
        start = 0 
        prediction_pred = []
        prediction_gt   = []
        
        ########Need To be Fixed : To precisely match size 
        while(start + self.batch_size < len(data_raw)):
            #print(start)
            key_list_batch  = key_list[start:start+self.batch_size]
            self._import_data(data_raw,key_list_batch,is_test=is_test)
            batch_labels_flat,batch_localizations_flat,batch_scores_flat=self._make_batch()
            feed_dict = {self.X:self.batch_imgs,self.Y_cls:batch_labels_flat,self.Y_loc:batch_localizations_flat,self.Y_score:batch_scores_flat,self.TRAINING:False}
            batch_bboxes_infer,batch_bboxes_cand,batch_cls_infer,batch_cls_score_infer=self._decode_from_logits(sess,data_raw,key_list_batch,is_test=is_test)
            self.nms(batch_bboxes_infer,batch_cls_infer,batch_cls_score_infer)
            batch_bboxes_test,bbox_candidate,batch_cls=self._decode_from_batch(data_raw,key_list_batch,is_test=is_test)

            
            for b in range(len(key_list_batch)):
                boxes_pred = self.batch_nms_bboxes[b]
                clses_pred = self.batch_nms_cls[b]
                boxes_gt   = batch_bboxes_test[b]
                clses_gt   = batch_cls[b]
                for i in range(len(boxes_pred)):
                    box_pred = boxes_pred[i]
                    cls_pred = clses_pred[i]
                    candidate = -1
                    max_jcd = 0
                    for j in range(len(boxes_gt)):
                        box_gt = boxes_gt[j]
                        jcd    = self.jaccard(box_pred,box_gt)
                        
                       
                        if jcd > 0.5 and jcd>max_jcd:
                            candidate = j
                            max_jcd = jcd
                    if candidate >= 0:
                        match_box_gt = boxes_gt[candidate]
                        match_cls_gt = clses_gt[candidate]
                        prediction_pred.append(cls_pred)
                        prediction_gt.append(match_cls_gt)
                    else:
                        prediction_pred.append(cls_pred)
                        prediction_gt.append(-1)
            start = start + self.batch_size    
        prediction_pred = np.array(prediction_pred)
        prediction_gt   = np.array(prediction_gt)
        acc = np.mean(prediction_pred==prediction_gt)
        print(acc)
        return acc
    def _generate_gt(self,sess,Y_cls,Y_loc,Y_score):
        self.gt_localization = sess.run()
    ###Data Augmentation###
    def _flip_images_and_boxes(self): # Horizonal flip
        batch_bboxes_flip = []
        batch_labels_flip = []
        batch_imgs_flip   = []
        for i in range(self.batch_size):
            img      = self.batch_imgs[i]
            img_flip = np.flip(img,axis=1)
            img_flip = np.array(img_flip)
            bboxes_flip = []
            for bbox in self.batch_bboxes[i]:
                bbox_flip = np.array([300-bbox[2],bbox[1],300-bbox[0],bbox[3]])
                bboxes_flip.append(bbox_flip)
            batch_bboxes_flip.append(bboxes_flip)
            batch_imgs_flip.append(img_flip)
            batch_labels_flip.append(self.batch_labels[i])
        self.batch_bboxes = batch_bboxes_flip
        self.batch_labels = self.batch_labels
        self.batch_imgs   = batch_imgs_flip
    #def 
    def smooth_L1(self,x):  
        absx  = tf.abs(x)
        minx=tf.minimum(absx,1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r 
    def jaccard(self,box1,box2):
            xmin1 = box1[0]
            ymin1 = box1[1]
            xmax1 = box1[2]
            ymax1 = box1[3]
            
            xmin2 = box2[0]
            ymin2 = box2[1]
            xmax2 = box2[2]
            ymax2 = box2[3]
            
            xmin_inter = np.maximum(xmin1,xmin2)
            ymin_inter = np.maximum(ymin1,ymin2)
            xmax_inter = np.minimum(xmax1,xmax2)
            ymax_inter = np.minimum(ymax1,ymax2)
            if (xmax_inter-xmin_inter)<0  or (ymax_inter-ymin_inter) < 0:
               return 0
            inter_volume = (xmax_inter-xmin_inter)*(ymax_inter-ymin_inter)
            union_volume = (xmax1-xmin1)*(ymax1-ymin1)+(xmax2-xmin2)*(ymax2-ymin2)-inter_volume
            jaccard = inter_volume/union_volume
            return jaccard   

class Transform():
    def __init__(self):
        pass
    def _new_batch(self,imgs,box_case):
        self.imgs = np.copy(imgs)
        self.box_case = []
        for boxes in box_case:
            boxes2 = np.copy(boxes)
            self.box_case.append(boxes2)
    def _brightness(self):
        dice  = np.random.random()> 0.5
        if dice:
            delta = np.random.random()/5-0.1 # delta in [-0.1, 0.1]
            imgs2 = self.imgs + delta
            imgs2[imgs2>1] = 1
            imgs2[imgs2<0] = 0
            self.imgs = imgs2
        pass
    def _contrast(self):
        dice  = np.random.random()> 0.5
        if dice:
            delta = np.random.random()+0.5 # delta in [0.5, 1.5]
            imgs2 = self.imgs * delta
            imgs2[imgs2>1] = 1
            imgs2[imgs2<0] = 0
            self.imgs = imgs2
        pass
    def _hue(self):
        dice  = np.random.random()> 0.5
        if dice:
            imgs2  = []
            delta = (np.random.random()-0.5)*36 # delta in [-18, 18]
            for img in self.imgs:
                img2 = img*255
                img2 = img2.astype(np.uint8)
                img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
                img3[:,:,0] = img3[:,:,0] + delta
                img3[:,:,0][img3[:,:,0]<0] += 180
                img3[:,:,0][img3[:,:,0]>180] -= 180
                img4 = cv2.cvtColor(img3,cv2.COLOR_HSV2BGR)
                img4 = img4 / 255
                imgs2.append(img4)
            self.imgs = np.array(imgs2)
        pass
    def _saturation(self):
        dice  = np.random.random()> 0.5
        if dice:
            imgs2  = []
            delta = np.random.random()+0.5 # delta in [0.5, 1.5]
            for img in self.imgs:
                img2 = img*255
                img2 = img2.astype(np.uint8)
                img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
                img3[:,:,1] = img3[:,:,1] * delta
                img3[:,:,1][img3[:,:,1]<0]   = 0 
                img3[:,:,1][img3[:,:,1]>255] = 255
                img4 = cv2.cvtColor(img3,cv2.COLOR_HSV2BGR)
                img4 = img4 / 255
                imgs2.append(img4)
            self.imgs = np.array(imgs2)
        pass 
    def _flip(self):
        dice  = np.random.random() > 0.5
        if dice:
            imgs2     = []
            box_case2 = []
            for img,boxes in zip(self.imgs,self.box_case):
                img_flip = np.flip(img,axis=1)
                boxes_flip = []
                for box in boxes:
                    box_flip = np.array([300-box[2],box[1],300-box[0],box[3]])
                    boxes_flip.append(box_flip)
                img_flip = np.array(img_flip)
                imgs2.append(img_flip)
                box_case2.append(np.array(boxes_flip))
            self.imgs = np.array(imgs2)
            self.box_case = box_case2
        pass
    def _expand(self):
        #dice  = np.random.random()< 0.25
        dice = True
        if dice:
            imgs2     = []
            box_case2 = []
            expand_ratio = (np.random.rand()+1.0)
            
            for img,boxes in zip(self.imgs,self.box_case):
                new_shape = int(img.shape[0]*expand_ratio)
                img2 = np.zeros((new_shape,new_shape,3))
                mean_value = np.array([104,117,123])/255
                img2[:,:] = mean_value
                start_w = int(np.random.random()*(expand_ratio-1)*img.shape[0])
                start_h = int(np.random.random()*(expand_ratio-1)*img.shape[1])
                img2[start_h:start_h+img.shape[0],start_w:start_w+img.shape[1],:] = img
                boxes2 = []
                for box in boxes:
                    box[0] = int(( box[0]+start_w) / new_shape * img.shape[0]  ) 
                    box[1] = int(( box[1]+start_h) / new_shape * img.shape[0]  )
                    box[2] = int(( box[2]+start_w) / new_shape * img.shape[0]  )
                    box[3] = int(( box[3]+start_h) / new_shape * img.shape[0]  )
                    boxes2.append(box)
                img2 = cv2.resize(img2,(300,300))
                
                imgs2.append(img2)
                box_case2.append(np.array(boxes2))
            self.imgs = np.array(imgs2)
            self.box_case = np.copy(box_case2)
        pass
    def _randcrop(self):
        crop_ratio = 1.0 - np.random.rand()/5
        #crop_ratio = 1.0 - np.random.rand()/1000
        new_length = int(300*crop_ratio)
        imgs2     = []
        box_case2 = []
        for img,boxes in zip(self.imgs,self.box_case):
            out = False
            start_w = int(np.random.random()*(1-crop_ratio)*img.shape[0])
            start_h = int(np.random.random()*(1-crop_ratio)*img.shape[1])
            cropped_figure = [start_h,start_w,start_h+new_length,start_w+new_length]
            for box in boxes:
                jcd = self.jaccard(cropped_figure,box)
                if box[0]-start_w < 0 or box[1]-start_h < 0 or box[2]>start_w+new_length or box[3]>start_h+new_length:
                
                    out = True
            if out:
               imgs2.append(img)
               box_case2.append(boxes)
               #print('out')
               continue
            
            img2    = img[start_h:start_h+new_length,start_w:start_w+new_length]
            img2    = cv2.resize(img2,(300,300))
            
            boxes2 = []
            for box in boxes:
                box[0] = int(( box[0]-start_w) / new_length * img.shape[0]  ) 
                box[1] = int(( box[1]-start_h) / new_length * img.shape[0]  ) 
                box[2] = int(( box[2]-start_w) / new_length * img.shape[0]  ) 
                box[3] = int(( box[3]-start_h) / new_length * img.shape[0]  ) 
                boxes2.append(box)
            imgs2.append(img2)
            box_case2.append(np.array(boxes2))
        self.imgs     = np.array(imgs2)
        self.box_case = np.copy(box_case2)
        pass
    def jaccard(self,box1,box2):
            xmin1 = box1[0]
            ymin1 = box1[1]
            xmax1 = box1[2]
            ymax1 = box1[3]
            
            xmin2 = box2[0]
            ymin2 = box2[1]
            xmax2 = box2[2]
            ymax2 = box2[3]
            
            xmin_inter = np.maximum(xmin1,xmin2)
            ymin_inter = np.maximum(ymin1,ymin2)
            xmax_inter = np.minimum(xmax1,xmax2)
            ymax_inter = np.minimum(ymax1,ymax2)
            
            if (xmax_inter-xmin_inter)<0  or (ymax_inter-ymin_inter) < 0:
                return 0
            inter_volume = (xmax_inter-xmin_inter)*(ymax_inter-ymin_inter)
            union_volume = (xmax1-xmin1)*(ymax1-ymin1)+(xmax2-xmin2)*(ymax2-ymin2)-inter_volume
            jaccard = inter_volume/union_volume
            return jaccard       

class_dict={1: 'cow',
 2: 'chair',
 3: 'bus',
 4: 'dog',
 5: 'person',
 6: 'pottedplant',
 7: 'car',
 8: 'boat',
 9: 'cat',
 10: 'bicycle',
 11: 'train',
 12: 'bottle',
 13: 'sofa',
 14: 'horse',
 15: 'aeroplane',
 16: 'diningtable',
 17: 'bird',
 18: 'motorbike',
 19: 'tvmonitor',
 20: 'sheep'}
 
flatten_len_list  = [5776,2166,600,150,36,16]