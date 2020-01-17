# 此文件用于创建网络主体结构

# coding:utf-8
# Bin GAO

import os
import tensorflow as tf
import numpy as np
from network import*

def conv2d(x,  kernel_size, n_filters,training, name, is_bn=False, activation=tf.nn.relu):
    """
    Args:
    x: A tensor of size [batch_size, height_in, width_in,channel_in]
    kernel_size: the size of the conv kernel
    n_filters: the number of the conv kernel
    training: if training,the variable is trainable
    is_bn:if to use batch_normalization this layer
    activation:if to use relu this layer
    Returns:
    A tensor of size [batch_size, height_out, width_out,channel_out]
    """
    with tf.variable_scope('layer{}'.format(name)):  
        for index, filter in enumerate(n_filters):  
            conv = tf.layers.conv2d(x, filter, kernel_size, strides=1, padding='same',trainable=training,
                                    name='conv_{}'.format(index + 1),
                                    )
            
            if is_bn != False:
                conv = tf.layers.batch_normalization(conv, training=training, name='bn_{}'.format(
                    index + 1))  

        if activation == None:  
            return conv

        conv = activation(conv, name='relu{}_{}'.format(name, index + 1))  

        return conv

def pool2d(x, pool_size, pool_stride, name):
    """
    Args:
    x: A tensor of size [batch_size, height_in, width_in,channel_in]
    pool_size: the size of the maxpooling kernel,with a shape[height，weight]
    pool_stride: the stride of the maxpooling,with a shape[stride1，stride2]
    Returns:
    A tensor of size [batch_size, height_out, width_out,channel_out]
    """
    pool = tf.layers.max_pooling2d(x, pool_size, pool_stride, name='pool_{}'.format(name), padding='same')
    return pool


def get_bilinear_filter(filter_shape, upscale_factor, name):
    """
    Args:
    x: A tensor of size [kernel_size, kernel_size, channels_out, channels_in]
    upscale_factor: the upscal factor for the deconv
    Returns:
    A tensor of size [height, width, channel_out, channel_in]
    """
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(initializer=init, name=name,
                                       shape=weights.shape, trainable=False)  
    return bilinear_weights


def deconv2d(bottom, upscale_factor, name, output_shape, n_channels=1):
    """
    Args:
    bottom: A tensor of size [batch_size, height_in, width_in,channel_in]
    upscale_factor: the upscal factor for the deconv
    output_shape: the shape of the output[batch_size, height_out, width_out,channel_out]
    Returns:
    A tensor of size [batch_size, height_out, width_out,channel_out]
    """
    kernel_size = 2 * upscale_factor 
    stride = upscale_factor
    strides = [1, stride, stride, 1]

    filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

    weights = get_bilinear_filter(filter_shape, upscale_factor, name)
    deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                    strides=strides, padding='SAME')

    return deconv



def unet(input,batch_size,h,w, training):
    """
    Args:
    input: the input of the network,a tensor of size [batch_size, height, width,channel]
    Returns:
    the output of sidelayer5,sidelayer4,sidelayer3,sidelayer2,sidelayer1 and the output of the whole network
    """
    #normoralize the input to[-1,1]
    image_R = input[:,:,:, 0:1]    
    image_G = input[:,:,:, 1:2] 
    image_B = input[:,:,:, 2:3]
    max_R=tf.reduce_max(image_R)
    min_R=tf.reduce_min(image_R)
    max_G=tf.reduce_max(image_G)
    min_G=tf.reduce_min(image_G)
    max_B=tf.reduce_max(image_B)
    min_B=tf.reduce_min(image_B)
    input_1 = 2*(image_R-min_R)/(max_R-min_R)-1
    input_2 = 2*(image_G-min_G)/(max_G-min_G)-1
    input_3 = 2*(image_B-min_B)/(max_B-min_B)-1
    input = tf.concat([input_1, input_2,input_3], axis=-1,name='inputs')
    
    conv1, conv2, conv3, conv4, conv5 = network(input,training)
    #sidelayer5
    conv1_dsn5 = conv2d(conv5, (7, 7), [256],training, name='conv1_dsn5',is_bn=False)  
    conv2_dsn5 = conv2d(conv1_dsn5, (7, 7), [256], training, name='conv2-dsn5',is_bn=False)
    conv3_dsn5 = conv2d(conv2_dsn5, (1, 1), [1], training, name='conv3-dsn5', activation=None)  
    score_dsn5_up = deconv2d(conv3_dsn5, 32, name='upsample16_in_dsn6_sigmoid-dsn5',output_shape=[batch_size, h, w, 1])   
    
    
    #sidelayer4
    conv1_dsn4 = conv2d(conv4, (5, 5), [256], training,name='conv1-dsn4',is_bn=False)  
    conv2_dsn4 = conv2d(conv1_dsn4, (5, 5), [256], training, name='conv2-dsn4',is_bn=False)
    conv3_dsn4 = conv2d(conv2_dsn4, (1, 1), [1], training, name='conv3-dsn4',activation=None) 
    score_dsn4_up = deconv2d(conv3_dsn4, 16, name='upsample8_in_dsn4_sigmoid-dsn4',output_shape=[batch_size, h, w, 1]) 
    
    
    #sidelayer3
    conv1_dsn3 = conv2d(conv3, (5, 5), [128], training,name='conv1-dsn3',is_bn=False)  
    conv2_dsn3 = conv2d(conv1_dsn3, (5, 5), [128], training, name='conv2-dsn3',is_bn=False)
    conv3_dsn3 = conv2d(conv2_dsn3, (1, 1), [1], training, name='conv3-dsn3', activation=None)
    score_dsn5_up_3 = deconv2d(conv3_dsn5, 4, name='upsample8_dsn6',output_shape=conv3_dsn3.get_shape().as_list()) 
    score_dsn4_up_3 = deconv2d(conv3_dsn4, 2, name='upsample4_dsn5', output_shape=conv3_dsn3.get_shape().as_list())
    concat_dsn3 = tf.concat([score_dsn5_up_3, score_dsn4_up_3, conv3_dsn3], axis=-1,name='concat_dsn3')  
    conv4_dsn3 = conv2d(concat_dsn3, (1, 1), [1], training, name='conv4-dsn3', activation=None)  
    score_dsn3_up = deconv2d(conv4_dsn3, 8, name='upsample4_in_dsn3_sigmoid-dsn3',output_shape=[batch_size, h, w, 1])  


    #sidelayer2
    conv1_dsn2 = conv2d(conv2, (5, 5), [128], training,name='conv1-dsn2',is_bn=False)  
    conv2_dsn2 = conv2d(conv1_dsn2, (5, 5), [128], training, name='conv2-dsn2',is_bn=False)
    conv3_dsn2 = conv2d(conv2_dsn2, (1, 1), [1], training, name='conv3-dsn2', activation=None)
    score_dsn5_up_2 = deconv2d(conv3_dsn5, 8, name='upsample8_dsn5', output_shape=conv3_dsn2.get_shape().as_list())
    score_dsn4_up_2 = deconv2d(conv3_dsn4, 4, name='upsample4_dsn4', output_shape=conv3_dsn2.get_shape().as_list())
    concat_dsn2 = tf.concat([ score_dsn5_up_2, score_dsn4_up_2, conv3_dsn2], axis=-1,name='concat_dsn2')  
    conv4_dsn2 = conv2d(concat_dsn2, (1, 1), [1], training, name='conv4-dsn2', activation=None)  
    score_dsn2_up = deconv2d(conv4_dsn2, 4, name='upsample2_in_dsn2_sigmoid-dsn2',output_shape=[batch_size, h, w, 1])  
    
    
    #sidelayer1
    conv1_dsn1 = conv2d(conv1, (3, 3), [64], training,name='conv1-dsn1',is_bn=False)
    conv2_dsn1 = conv2d(conv1_dsn1, (3, 3), [64], training, name='conv2-dsn1',is_bn=False)
    conv3_dsn1 = conv2d(conv2_dsn1, (1, 1), [1], training, name='conv3-dsn1', activation=None)
    score_dsn5_up_1 = deconv2d(conv3_dsn5, 16, name='upsample16_dsn5', output_shape=conv3_dsn1.get_shape().as_list())
    score_dsn4_up_1 = deconv2d(conv3_dsn4, 8, name='upsample8_dsn4', output_shape=conv3_dsn1.get_shape().as_list())
    score_dsn3_up_1 = deconv2d(conv3_dsn3, 4, name='upsample4_dsn3', output_shape=conv3_dsn1.get_shape().as_list())
    score_dsn2_up_1 = deconv2d(conv3_dsn2, 2, name='upsample2_dsn2', output_shape=conv3_dsn1.get_shape().as_list())
    concat_dsn1 = tf.concat([ score_dsn5_up_1, score_dsn4_up_1, score_dsn3_up_1,score_dsn2_up_1, conv3_dsn1], axis=-1,name='concat_dsn1')
    score_dsn1_up = conv2d(concat_dsn1, (1, 1), [1], training, name='conv4-dsn1',activation=None)  
    score_dsn1_up = deconv2d(score_dsn1_up, 2, name='upsample1_in_dsn1_sigmoid-dsn1',output_shape=[batch_size, h, w, 1])  

    
    concat_upscore = tf.concat([ score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up],axis=-1, name='concat')

    
    #the final output of the network
    upscore_fuse = tf.layers.conv2d(concat_upscore, filters=1, kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                                    name='output1', kernel_initializer=tf.constant_initializer(
            0.2))  

    return  score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse  

def loss_CE(y_pred, y_true):
    """
    Args:
    y_pred: the prediction of the input
    y_true: the ground truth of the input
    Returns:
    the sigmoid_cross_entropy_with_logits loss between the prediction and ground truth
    """
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)  
    cross_entropy_mean = tf.reduce_mean(cross_entropy) 
    return cross_entropy_mean



def train_op(loss, learning_rate):
    global_step = tf.train.get_or_create_global_step()  
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    return optimizer.minimize(loss, global_step=global_step) 



