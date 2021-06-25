#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 5/16/2020 8:36 PM 
# @Author : Zhicheng Zhang 
# @E-mail : zhicheng0623@gmail.com
# @Site :  
# @File : Train.py 
# @Software: PyCharm

import datetime, time
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('tkagg')
import tensorflow as tf
import numpy as np
import os
import sys
import glob
import pydicom
import random
import shutil
from skimage.measure import compare_mse as MSE
from skimage.measure import compare_ssim as ssim
import copy
import subprocess
import tensorflow.contrib as tf_contrib
sys.path.append("./utils")

import utils.ckpt as ckpt
import utils.loss as loss
# from ops import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.set_random_seed(0)

class LDCTNet(object):
    def __init__(self):
        super(LDCTNet, self).__init__()

        self.param = {}

        self.param['nx'] = 512
        self.param['ny'] = 512

        self.param['numIterations'] = 50000
        self.param['u_water'] = 0.0205
        self.param['batchsize'] = 8

        self.param['lr'] = 0.00005
        self.param['retrain'] = False
        self.param['epoch']  = 2500
        self.param['txt_save_path']         = os.path.join('./txt'      ,os.path.basename(__file__).split('.')[0])
        self.param['figure_save_path']      = os.path.join('./Figure'   ,os.path.basename(__file__).split('.')[0])
        self.param['model_save_path']       = os.path.join('./Results'  ,os.path.basename(__file__).split('.')[0])
        self.param['tensorboard_save_logs'] = os.path.join('./logs'     ,os.path.basename(__file__).split('.')[0])
        self.wint = tf.contrib.layers.xavier_initializer() # tf.random_normal_initializer(0, 0.01)  #
        self.regularizer = None#tf.contrib.layers.l2_regularizer(scale=0.01)


    def scaled_dot_product_attention(self,Q, K, V, name="scaled_dot_product_attention"):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, K, transpose_b=True)  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            outputs = tf.nn.softmax(outputs)


            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs


    def multihead_attention(self,queries, keys, values, num_heads=8, name="MHSA"):
        d_model = queries.get_shape().as_list()[-1]   #
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            # Linear projections
            Q = tf.layers.dense(queries,    queries.get_shape().as_list()[-1] ,  use_bias=True,name='linear_QK', reuse=False, kernel_regularizer=self.regularizer,kernel_initializer=self.wint)
            K = tf.layers.dense(keys,       keys.get_shape().as_list()[-1] ,     use_bias=True,name='linear_QK', reuse=True, kernel_regularizer=self.regularizer,kernel_initializer=self.wint)
            V = tf.layers.dense(values,     values.get_shape().as_list()[-1],    use_bias=True,name='linear_V',  reuse=False, kernel_regularizer=self.regularizer,kernel_initializer=self.wint)
            
            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, 512/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, 512/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, 512/h)

            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, 512)

            # Linear projections
            outputs = tf.layers.dense(outputs, d_model, kernel_regularizer=self.regularizer,kernel_initializer=self.wint)
        return outputs
 
    def ff(self,inputs, num_units, name="FFN"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            # Inner layer
            outputs = tf.layers.dense(inputs,  num_units[0], activation=tf.nn.leaky_relu, kernel_regularizer=self.regularizer,kernel_initializer=self.wint)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1], kernel_regularizer=self.regularizer,kernel_initializer=self.wint)
               
        return outputs
    def Encode_layer(self, input, n_heads = 8, name = 'EncoderLayer'):
        b, wh, c = input.get_shape().as_list()
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            ###############################################################################################

            x = self.multihead_attention(input, input, input, num_heads=n_heads, name="MHSA1")
            x += input

            out = self.ff(x, [8*c,c], name="ff1")
            x += out

            return x

    def Decode_layer(self, input, memory, n_heads = 8, name = 'Decode_layer'):
        b, wh, c = input.get_shape().as_list()
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            ###############################################################################################
            x = self.multihead_attention(input, input, input, num_heads=n_heads, name="MHSA1")
            x += input

            out = self.multihead_attention(x, memory, memory, num_heads=n_heads, name="MHSA2")
            x += out

            out = self.ff(x, [8*c,c], name="ff1")
            x += out

            return x
    def model(self, input,name='model'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            window = loss._tf_fspecial_gauss(11, 1.5)  # window shape [size, size]  # 357 before  1.5
            img_LR = tf.nn.conv2d(input, window, strides=[1, 1, 1, 1], padding='SAME') 
            tf.summary.image('img_LR',      tf.clip_by_value(img_LR, 0.0,  0.03),max_outputs=1)

            img_HR = input - img_LR

            ##############################################################################################################
            ch = 16
            with tf.variable_scope('LR', reuse=tf.AUTO_REUSE):
                x = img_LR
                
                x_256 = tf.layers.conv2d(x,     filters=16,  kernel_size=[5,5], strides=[2,2], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer,activation=tf.nn.leaky_relu) 
                x_128 = tf.layers.conv2d(x_256, filters=32,  kernel_size=[5,5], strides=[2,2], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer,activation=tf.nn.leaky_relu) 
                
                x_64_lr = tf.layers.conv2d(x_128,  filters=64,  kernel_size=[5,5], strides=[2,2], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer,activation=tf.nn.leaky_relu) 
                x_32_lr = tf.layers.conv2d(x_64_lr,   filters=256, kernel_size=[5,5], strides=[2,2], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer,activation=tf.nn.leaky_relu) 
                     
                
                x_64_hr = tf.layers.conv2d(x_128,  filters=64,  kernel_size=[5,5], strides=[2,2], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer,activation=tf.nn.leaky_relu) 
                x_32_hr = tf.layers.conv2d(x_64_hr,   filters=128, kernel_size=[5,5], strides=[2,2], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer,activation=tf.nn.leaky_relu) 
                x    = tf.layers.conv2d(x_32_hr,   filters=256, kernel_size=[5,5], strides=[2,2], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer,activation=tf.nn.leaky_relu) 
  
                memory = tf.reshape(x,[x.get_shape().as_list()[0],-1, x.get_shape().as_list()[-1]]) #b,16*16, 256

                for i in range(3):
                    memory = self.Encode_layer(memory, n_heads = 8, name = 'Encode_layer'+str(i))
            ##############################################################################################################
            with tf.variable_scope('HR', reuse=tf.AUTO_REUSE):
                img_HR_patch = tf.space_to_depth(img_HR, block_size=16)
                x = img_HR_patch
                for i in range(3):
                    x = tf.layers.conv2d(x,   filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer) 
                    x = tf.nn.leaky_relu(x)

                b,w,h,c = x.get_shape().as_list()
                x = tf.reshape(x,[b,-1, c])  #b,32*32, 256

                for i in range(3):
                    x = self.Decode_layer(x, memory, n_heads = 8, name = 'Decode_layer'+str(i))

                x = tf.reshape(x,[b,w,h,c] )#b,32, 32, 256

                

            with tf.variable_scope('Combine', reuse=tf.AUTO_REUSE):
                fea_32 = x + x_32_lr
                x = fea_32
                for i in range(2):
                    x = tf.layers.conv2d(x,   filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer) 
                    x = tf.nn.leaky_relu(x)
                x += fea_32

                x_hr = tf.depth_to_space(x, block_size=2)

                fea_64 = x_hr + x_64_lr
                x = fea_64
                for i in range(2):
                    x = tf.layers.conv2d(x,   filters=64, kernel_size=[3,3], strides=[1,1], padding='SAME',kernel_initializer=self.wint, kernel_regularizer=self.regularizer) 
                    x = tf.nn.leaky_relu(x)

                x = tf.nn.relu(x + fea_64)

                out = tf.depth_to_space(x, block_size=8)

        return out

    def build_model(self):
        self.img_syn = self.model(self.img_in, name='model')

        g_loss_mse = tf.losses.mean_squared_error(self.img_syn,self.img_org)      
        g_loss_init = g_loss_mse 

        global_step = tf.Variable(0)
        g_ADAM_init = tf.train.AdamOptimizer(self.lr).minimize(g_loss_init,global_step=global_step)



        return g_ADAM_init

    def train(self):
        checkpointdir = self.param['model_save_path']
        ####################################################################################

        patient_list_train = ['L067','L096','L143','L192','L286','L291','L310']
        patient_list_valid = ['L333']
        patient_list_test  = ['L506','L109']

        ####################################################################################
        if not os.path.exists(checkpointdir):
            checkpoints_dir = checkpointdir
            os.makedirs(checkpoints_dir)
        else:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            checkpoints_dir = checkpointdir.format(current_time)
        ####################################################################################

        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            ####################################################################################
            self.img_in    = tf.placeholder(tf.float32,[self.param['batchsize'],     self.param['nx'],    self.param['ny'],1])
            self.img_org   = tf.placeholder(tf.float32,[self.param['batchsize'],     self.param['nx'],    self.param['ny'],1])
            self.sigma     = tf.placeholder(tf.float32)
            self.lr        = tf.placeholder(tf.float32)

            g_ADAM_init = self.build_model()

            ####################################################################################

            with tf.Session(config=config, graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                ####################################################################################
                # Read the existed model
                if not self.param['retrain']:
                    checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                    meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    ckpt.load_ckpt(sess=sess, save_dir=checkpoints_dir, is_latest=True,var_list=[var for var in tf.trainable_variables()])
                    epoch_pre = int(meta_graph_path.split("-")[1].split(".")[0])
                else:
                    sess.run(tf.global_variables_initializer())
                    epoch_pre = 0

                ####################################################################################
                merged_summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(self.param['tensorboard_save_logs'], sess.graph)
                ####################################################################################
                if os.path.exists(os.path.join(self.param['txt_save_path'],'valid_results_rmse.txt')) and self.param['retrain']:
                    os.remove(os.path.join(self.param['txt_save_path'],'valid_results_rmse.txt'))
                if os.path.exists(os.path.join(self.param['txt_save_path'],'valid_results_ssim.txt')) and self.param['retrain']:
                    os.remove(os.path.join(self.param['txt_save_path'],'valid_results_ssim.txt'))
                ####################################################################################
                img_LD_all = np.zeros(shape=[4482,512,512,1])
                img_ND_all = np.zeros(shape=[4482,512,512,1])
                slice_id = 0
                for i in range(7):
                    case_list_LD = sorted(glob.glob(os.path.join('/home/zhang/Dataset/Mayo low dose CT/Image_1mm',patient_list_train[i],'quarter_1mm/*.IMA')))
                    case_num_LD = np.shape(case_list_LD)[0]
                    case_list_ND = sorted(glob.glob(os.path.join('/home/zhang/Dataset/Mayo low dose CT/Image_1mm',patient_list_train[i],'full_1mm/*.IMA')))
                    for j in range(0,case_num_LD):
                        filename =  case_list_LD[j]
                        img = pydicom.dcmread(filename)
                        img_LD = ((img.pixel_array - 1000.) / 1000.0) * self.param['u_water'] + self.param['u_water']
                        img_LD[img_LD < 0] = 0
                        img_LD_all[slice_id,:,:,0] = img_LD

                        filename = case_list_ND[j]
                        img = pydicom.dcmread(filename)
                        img_ND = ((img.pixel_array - 1000.) / 1000.0) * self.param['u_water'] + self.param['u_water']
                        img_ND[img_ND < 0] = 0
                        img_ND_all[slice_id,:,:,0] = img_ND  
                        slice_id += 1
                if ~self.param['retrain']:
                    epoch_pre += 1
                for epoch in range(epoch_pre,epoch_pre+300):
                    Train_Num = [i for i in range(4482)]
                    random.shuffle(Train_Num)
                    if epoch <= 180:
                        learnrate = 0.0001                 
                    else:
                        learnrate = 0.00001

                    for batch_id in range(0,4482//self.param['batchsize']):
                        sess.run(g_ADAM_init,feed_dict={self.img_in:  img_LD_all[Train_Num[batch_id*self.param['batchsize']:(1+batch_id)*self.param['batchsize']],:,:,:],
                                                        self.img_org: img_ND_all[Train_Num[batch_id*self.param['batchsize']:(1+batch_id)*self.param['batchsize']],:,:,:]，self.lr : learnrate})
                    if epoch % 1 == 0:
                        var_list = [var for var in tf.trainable_variables()]
                        print(time.asctime(time.localtime(time.time())),'the %d-th iterations. Saving Models...' % epoch)
                        ckpt.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir=checkpoints_dir,global_step=epoch, var_list=var_list)
                        print("[*] Saving checkpoints SUCCESS! Begin the validation stage")

if __name__ == '__main__':
    srcnn = LDCTNet()
    srcnn.train()
