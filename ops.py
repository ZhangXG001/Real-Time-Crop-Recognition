import tensorflow as tf
import tensorflow.contrib as tf_contrib


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init =tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf_contrib.layers.l2_regularizer(0.005)


##################################################################################
# Layer
##################################################################################

def conv(x,channels,is_train,name,kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x,filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             strides=stride, use_bias=use_bias, padding=padding,trainable=is_train)

        return x

def resblock(x_init, channels, training, first=False,use_bias=True, downsample=False, scope='bottle_resblock') :
    """
    Args:
    x_init: a tensor of size [batch_size, height,width,channel]
    channels: the number of the filters
    training: if training,the variable is trainable
    first: if the first conv of the block
    use_bias: if to use bias 
    downsample: if downsample,stride=2
    Returns:
    A tensor of size [batch_size, height, width,channel]
    """
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, training)
        x = relu(x)

        if downsample :
            x = conv(x, channels,training, name='block/conv1',kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            if first==True:
                x_init = conv(x_init, channels,training,name='block/conv11', kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
        else :
            x = conv(x, channels,training,name='block/conv1', kernel=1, stride=1, use_bias=use_bias, scope='conv_0')
            if first == True:
               x_init = conv(x_init, channels,training,name='block/conv11', kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, training)
        x = relu(x)
        x = conv(x, channels, training,name='block/conv2',kernel=3, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        return x + x_init
    
def bottle_resblock(x_init, channels, training, first=False,use_bias=True, downsample=False, scope='bottle_resblock') :
    """
    Args:
    x_init: a tensor of size [batch_size, height,width,channel]
    channels: the number of the filters
    training: if training,the variable is trainable
    first: if the first conv of the block
    use_bias: if to use bias 
    downsample: if downsample,stride=2
    Returns:
    A tensor of size [batch_size, height, width,channel]
    """
    with tf.variable_scope(scope) :
        #shortcut=x_init
        x = batch_norm(x_init, training)
        shortcut = relu(x)

        if downsample :
            x = conv(shortcut, channels,training,name='block/conv1', kernel=1, stride=2, use_bias=use_bias, scope='conv_0')
            if first==True:
                shortcut = conv(shortcut, channels*4,training,name='block/conv2', kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
        else :
            x = conv(shortcut, channels,training,name='block/conv1', kernel=1, stride=1, use_bias=use_bias, scope='conv_0')
            if first == True:
                shortcut = conv(shortcut, channels*4,training,name='block/conv2', kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, training)
        x = relu(x)
        x = conv(x, channels, training,name='block/conv3',kernel=3, stride=1, use_bias=use_bias, scope='conv_1x1_front')

        x = batch_norm(x, training)
        x = relu(x) 
        x = conv(x, channels*4,training,name='block/conv3', kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []
    if res_n == 10 :
        x = [1, 1, 1, 1]
    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################

# Normalization function
##################################################################################
    
def batch_norm(x, training,momentum=0.9):
    return  tf.layers.batch_normalization(x, training=training, momentum=momentum)
##################################################################################

def pool(layer_name, x, ksize=None, stride=None, is_max_pool=True):
    
    ksize = ksize if ksize else [1, 2, 2, 1]
    stride = stride if stride else [1, 2, 2, 1]

    if is_max_pool:
        x = tf.nn.max_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, ksize, strides=stride, padding='SAME', name=layer_name)

    return x


