import time
from ops import *
#from utils import *

def network( x,training,res_n=10, reuse=False):
    """
    Args:
    x: the input of the network,a tensor of size [batch_size, height,width,channel]
    training: if training
    res_n: the number of the blocks to use for each layer
    Returns:
    the output of each layer(conv1,conv2,conv3,conv4,conv5)
    """
    with tf.variable_scope("network", reuse=reuse):

        if res_n < 50:
            residual_block = resblock
        else:
            residual_block = bottle_resblock

        residual_list = get_residual_layer(res_n)

        ch = 64  # paper is 64
        x = conv(x, ch,training,name='block/conv0',kernel=7, stride=2, scope='conv')
        conv1 = relu(x)
        print(conv1)
        print(x)
        x=pool('maxpooling', x, ksize=None, stride=None, is_max_pool=True)
        ########################################################################################################
        x = residual_block(x, ch ,training,first=True,  downsample=False, scope='resblock0_0')
        print(x)
        for i in range(1,residual_list[0]):
            x = residual_block(x,ch, training, downsample=False, scope='resblock0_' + str(i))
            print(x)
        conv2 = relu(x)
        print(conv2)
        ########################################################################################################
        x = residual_block(x, ch * 2,training,first=True,  downsample=True, scope='resblock1_0')
        print(x)
        for i in range(1, residual_list[1]):
            x = residual_block(x, ch * 2, training, downsample=False,
                               scope='resblock1_' + str(i))
            print(x)

        conv3 =relu(x)
        print(conv3)
        ########################################################################################################
        x = residual_block(x, ch * 4,training,first=True,  downsample=True, scope='resblock2_0')
        print(x)
        for i in range(1, residual_list[2]):
            x = residual_block(x,ch * 4, training, downsample=False,
                               scope='resblock2_' + str(i))
            print(x)
        conv4 =relu(x)
        print(conv4)
        ########################################################################################################
        x = residual_block(x,ch * 8,training,first=True,  downsample=True, scope='resblock3_0')
        print(x)

        for i in range(1,residual_list[3]):
            x = residual_block(x, ch * 8, training, downsample=False,
                               scope='resblock_3_' + str(i))
            print(x)
        conv5 =relu(x)
        print(conv5)
        ########################################################################################################

        return conv1,conv2,conv3,conv4,conv5
