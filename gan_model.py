# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:38:07 2017

@author: 49603
"""
import tensorflow as tf 
import numpy as np
from cell import ConvLSTMCell
from sub_function import *
import os

os.environ["CUDA_VISIBLE_DEVICES"]='1'





output_height=27
output_width=27
c_dim=20#d的输入通道
'''z_dim=116#g的输入通道'''
z_dim=110#g的输入通道
'''gf_dim=16#g的输出通道
df_dim=16#d的输出通道'''
gf_dim=9#g的输出通道
df_dim=9#d的输出通道
batch_size=128
'''class_number=16'''
class_number=10
keep_prob=0.9

s_h, s_w =output_height, output_width#27*27
s_h2, s_w2 = conv_out_size_same(s_h,2), conv_out_size_same(s_w,2)#s_h/2向上取整14*14
s_h4, s_w4 = conv_out_size_same(s_h2,2), conv_out_size_same(s_w2,2)#7*7
s_h8, s_w8 = conv_out_size_same(s_h4,2), conv_out_size_same(s_w4,2)#4*4
s_h16, s_w16 = conv_out_size_same(s_h8,2), conv_out_size_same(s_w8,2)#2*2
s_h32, s_w32 = conv_out_size_same(s_h16,2), conv_out_size_same(s_w16,2)#1*1
#rnn+z_spe concat

def generator(z,z_sper,d0,d1,d2,d3):
    #scope.reuse_variables()
    g_bn0 = batch_norm(name='g_bn0')#定义变量
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')
    g_bn3 = batch_norm(name='g_bn3')
          
    # project `z` and reshape
    z1=linear(z, gf_dim*8*s_h16*s_w16-72,'g_h0_lin')#对输入噪声线性化输出形式为16*8*2*2=512,一维
    
    z_=tf.concat([z1,z_sper],axis=1)
    print(np.shape(z1),np.shape(z_))
        
    h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])#将线性化的z_重塑为1个2维形式2*2*128，128指输出通道，-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算
    h00 = tf.nn.relu(g_bn0(h0))#对数据正则化
    h000 = add(h00, d3)
    h0_rnn = cell2(h000,name="cell2")
    asa0=Squeeze_excitation_layer(h0_rnn, out_dim=df_dim*8, ratio=4, layer_name='squeeze_layer1')
    ag0 = NonLocalBlock(h0_rnn, output_channels=gf_dim*8, scope="Non-local1")
    chsp0=add(asa0,ag0)
    
    
    h1 = deconv2d(chsp0, [batch_size, s_h8, s_w8, gf_dim*4], gf_dim*8, gf_dim*4, name='g_h1')#h0输入图像，[卷积核个数，卷积核的高度，卷积核的宽度，图像通道数]，input_channel,output_channel
    h10 = tf.nn.relu(g_bn1(h1))
    h100 = add(h10, d2)
    asa1=Squeeze_excitation_layer(h100, out_dim=df_dim*4, ratio=4, layer_name='squeeze_layer2')
    ag1 = NonLocalBlock(h100, output_channels=gf_dim*4, scope="Non-local2")
    chsp1=add(asa1,ag1)
    
    h2 = deconv2d(chsp1, [batch_size,s_h4,s_w4,gf_dim*2],gf_dim*4,gf_dim*2, name='g_h2')
    h20 = tf.nn.relu(g_bn2(h2))
    h200 = add(h20, d1)
    asa2=Squeeze_excitation_layer(h200, out_dim=df_dim*2, ratio=4, layer_name='squeeze_layer3')
    ag2 = NonLocalBlock(h200, output_channels=gf_dim*2, scope="Non-local3")
    chsp2=add(asa2,ag2)
    
    
    h3 = deconv2d(chsp2, [batch_size,s_h2,s_w2,gf_dim],gf_dim*2,gf_dim, name='g_h3')
    h30 = tf.nn.relu(g_bn3(h3))
    h300 = add(h30, d0)
    asa3=Squeeze_excitation_layer(h300, out_dim=df_dim, ratio=4, layer_name='squeeze_layer4')
    ag3 = NonLocalBlock(h300, output_channels=gf_dim, scope="Non-local4")
    chsp3=add(asa3,ag3)
    
    h4 = deconv2d(chsp3, [batch_size,s_h, s_w, c_dim],gf_dim,c_dim, name='g_h4')

    return tf.nn.tanh(h4)

def discriminator(image):
    #with tf.variable_scope("discriminator"):
    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    d_bn3 = batch_norm(name='d_bn3')
    d_bn4 = batch_norm(name='d_bn4')
    
    #orri0=Squeeze_excitation_layer(image, out_dim=c_dim, ratio=4, layer_name='dsqueeze_layer0')
    h0 = tf.nn.relu(conv2d(image, c_dim, df_dim, name='d_h0_conv'))#input_, input_channel, output_channel(14*14*16)
    p0=tf.layers.max_pooling2d(h0,pool_size=2,strides=1,padding='SAME',name='pool0')
   
    #a0=Squeeze_excitation_layer(p0, out_dim=df_dim, ratio=4, layer_name='dsqueeze_layer1')
    #ag0 = NonLocalBlock(p0, output_channels=df_dim, scope="dNon-local1")
    #agspa0=add(a0,ag0)
    
    h1 = tf.nn.relu(d_bn1(conv2d(p0, df_dim,df_dim*2, name='d_h1_conv')))#(7*7*32)
    p1=tf.layers.max_pooling2d(h1,pool_size=2,strides=1,padding='SAME',name='pool1')
    
    
    #a1=Squeeze_excitation_layer(p1, out_dim=df_dim*2, ratio=4, layer_name='dsqueeze_layer2')
    #ag1 = NonLocalBlock(p1, output_channels=df_dim*2, scope="dNon-local2")
    #agspa1=add(a1,ag1)
    
    h2 = tf.nn.relu(d_bn2(conv2d(p1, df_dim*2,df_dim*4, name='d_h2_conv')))#(4*4*64)
    p2=tf.layers.max_pooling2d(h2,pool_size=2,strides=1,padding='SAME',name='pool2')

    
    #a2=Squeeze_excitation_layer(p2, out_dim=df_dim*4, ratio=4, layer_name='dsqueeze_layer3')
    #ag2 = NonLocalBlock(p2, output_channels=df_dim*4, scope="dNon-local3")
    #agspa2=add(a2,ag2)
    
    h3 = tf.nn.relu(d_bn3(conv2d(p2, df_dim*4,df_dim*8, name='d_h3_conv')))#(2*2*128)
    p3=tf.layers.max_pooling2d(h3,pool_size=2,strides=1,padding='SAME',name='pool3')
    #ag3 = NonLocalBlock(p3, output_channels=df_dim*8, scope="dNon-local4")
    
    h3_rnn = cell1(p3,name="cell1")
    
    z_sper= tf.reshape(tf.nn.avg_pool(h3_rnn,[1,27,27,1],[1,27,27,1],padding='SAME'),[-1,h3_rnn.get_shape().as_list()[3]])
    #print(np.shape(z_sper))
    #print(h0,h1,h2,h3,s_w32)
    #h5 = tf.nn.relu(d_bn4(conv2d(h3, df_dim*8,df_dim*8, name='d_h5_conv')))
    #p3=tf.layers.max_pooling2d(h3,pool_size=2,strides=2,padding='SAME',name='pool3')
    h4= linear(tf.reshape(h3_rnn, [-1, df_dim*8*s_h16*s_w16]), class_number,'d_h3_lin')
    #h4= linear(tf.nn.dropout(tf.reshape(h3, [-1, df_dim*8*s_h16*s_w16]),0.5), class_number, 'd_h3_lin')
    
    return h4,z_sper,p0,p1,p2,p3