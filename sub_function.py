# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:27:38 2017

@author: 49603
"""

import math
import numpy as np 
import tensorflow as tf
from cell import ConvLSTMCell
import os
from tensorflow.python.layers.core import dense

os.environ["CUDA_VISIBLE_DEVICES"]='1'
def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))#将x向上取整作为整体返回为卷积层输出
  
  
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):#初始化:epsilon:防零极小值；momentum:滑动平均参数:name:节点名称
    with tf.variable_scope(name):#定义操作的上下文管理器，用于创建变量(层)。
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):#一个封装了的会在内部调用batch_normalization进行正则化的高级接口
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)#用于构建神经网络层、归一化，批处理等。



def conv2d(input_, input_channel, output_channel, d_h=2, d_w=2, k_h=5, k_w=5,  stddev=0.02,name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_channel,output_channel],initializer=tf.random_normal_initializer(stddev=stddev))
    #权重形式定义，大小[卷积核h,w,输入特征图数，输出特征图数]，值为生成标准正态分布的随机数
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')#卷积后的图与原图大小一致

    biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))#偏置形式定义，初始化值为常量0，个数对应于输出通道个数
    conv=tf.nn.bias_add(conv, biases)

    return conv

#def deconv2d(input_, output_shape,input_channel,output_channel,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name=None):
#  with tf.variable_scope(name):
#    # filter : [height, width, output_channels, in_channels]
#    w = tf.get_variable('w', [k_h, k_w, output_channel, input_channel],initializer=tf.random_normal_initializer(stddev=stddev))
#    
#    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
#    
#    biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
#    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#
#    return deconv
    
def deconv2d(input_, output_shape,input_channel,output_channel,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name=None):
    '''Deconv with resize'''
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
#    w = tf.get_variable('w', [k_h, k_w, output_channel, input_channel],initializer=tf.random_normal_initializer(stddev=stddev))
#    
#    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
#    
#    biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
#    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#        x = conv2d(input_,input_channel,output_channel,d_h=1, d_w=1, k_h = 5, k_w = 5 )
        x = tf.image.resize_nearest_neighbor(input_, output_shape[1:3], name = 'upsampling')
        x = conv2d(x,input_channel,output_channel,d_h=1, d_w=1, k_h=1, k_w=1 )
        return x
    
def cell1(input_,name="cell1"):
  with tf.variable_scope(name, reuse=None):
    aaa=input_.get_shape().as_list()
    conv_1=tf.reshape(input_,[-1,aaa[1],aaa[2],int(aaa[3]/8),8])#8是时刻，大小不要太大，会漫
    conv_11=tf.transpose(conv_1,perm=[0,4,1,2,3])
    #cell1 = ConvLSTMCell([aaa[1],aaa[2]],128, [4,4])
    cell1 = ConvLSTMCell([aaa[1],aaa[2]],72, [4,4])
    outputs1, state1 = tf.nn.dynamic_rnn(cell1, conv_11, dtype=tf.float32)
    return  state1[1]#o(t)  



def cell2(input_,name="cell2"):
  with tf.variable_scope(name, reuse=None):
    aaa=input_.get_shape().as_list()
    conv_2=tf.reshape(input_,[-1,aaa[1],aaa[2],int(aaa[3]/8),8])#8是时刻，大小不要太大，会漫
    conv_22=tf.transpose(conv_2,perm=[0,4,1,2,3])
    #cell2 = ConvLSTMCell([aaa[1],aaa[2]],128, [4,4])
    cell2 = ConvLSTMCell([aaa[1],aaa[2]],72, [4,4])
    outputs2, state2 = tf.nn.dynamic_rnn(cell2, conv_22, dtype=tf.float32)
    return  state2[1]#o(t)  


     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size,name=None,stddev=0.02, bias_start=0.0):#进行线性运算，获取一个随机正态分布矩阵，获取初始偏置值，如果with_w为真，则返回xw+b，权值w和偏置值b；否则返回xw+b。
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
      
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))#生成标准正态分布的随机数矩阵[特征图的高，输出的特征图个数]
      bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))
   
      return tf.matmul(input_, matrix) + bias

def cross_entropy(x, y):#求交叉熵
     
    return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
    
def selu(x):#缩放指数线性单元，根据该激活函数得到的网络具有自归一化功能。使用该激活函数后使得样本分布满足零均值和单位方差： 
  #with ops.name_scope('elu') as scope:
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale*tf.where(x>0.0,x,alpha*tf.exp(x)-alpha)

def concat(x,y):
    return tf.concat([x,y],3)
    
def add(x,y):
    return tf.add(x,y)


def concat3(x,y):
    return tf.concat([x,y],4)

def Fully_connected(x, units=16, layer_name='fully_connected') :
    with tf.variable_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return tf.keras.layers.GlobalAveragePooling2D()(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)



#[senet]:x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))
def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale
    
def NonLocalBlock(input_x, output_channels, sub_sample=False, is_bn=True, is_training=True, scope="NonLocalBlock"):
    batch_size, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope):
        with tf.variable_scope("g"):
            g = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="g_conv")
            if sub_sample:
                g = tf.layers.max_pooling2d(inputs=g, pool_size=2, strides=2, padding="valid", name="g_max_pool")
                print(g.shape)

        with tf.variable_scope("phi"):
            phi = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="phi_conv")
            if sub_sample:
                phi = tf.layers.max_pooling2d(inputs=phi, pool_size=2, strides=2, padding="valid", name="phi_max_pool")
                #print(phi.shape)

        with tf.variable_scope("theta"):
            theta = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="theta_conv")
            print(theta.shape)

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x = tf.reshape(g, [-1, height * width, output_channels])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x = tf.reshape(phi, [-1, output_channels, height * width])
        #print(phi_x.shape)

        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x = tf.reshape(theta, [-1, height * width, output_channels])
        print(theta_x.shape)

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)      
        y = tf.matmul(f_softmax, g_x)

        y = tf.reshape(y, [-1, height, width, output_channels])

        with tf.variable_scope("w"):
            w_y = tf.layers.conv2d(inputs=y, filters=in_channels, kernel_size=1, strides=1, padding="same", name="w_conv")
            if is_bn:
                w_y= tf.layers.batch_normalization(w_y, axis=3, training=is_training)    ### batch_normalization
        z = input_x + w_y

        return z