import os
import tensorflow as tf
import numpy as np
import argparse  
import pandas as pd  
import model
import time  
import losses as lossx
from model import train_op  
from model import loss_CE
from tensorflow.python.framework import graph_util
parser = argparse.ArgumentParser()
#you can optionly change the model_dir/logdir params
#the path of the train.csv file
parser.add_argument('--training_dir', 
                    default='./train.csv')  
#the path of the validation.csv file
parser.add_argument('--validation_dir', 
                    default='./validation.csv')
#the path of the model file
parser.add_argument('--model_dir',  
                    default='./model1')
#the path of the log file
parser.add_argument('--logdir', 
                    default='./logs1')
#the channel of input image
parser.add_argument('--c_image', 
                    type=int,
                    default=3)
#the channel of the label
parser.add_argument('--c_label', 
                    type=int,
                    default=1) 
#the height of the image
parser.add_argument('--h', 
                    type=int,
                    default=300)
#the width of the image
parser.add_argument('--w', 
                    type=int,
                    default=400) 
#batch size
parser.add_argument('--batch_size', 
                    type=int,
                    default=5)
#learning rate
parser.add_argument('--learning_rate',  
                    type=float,
                    default=1e-4)
#every decay_step to decay learning rate
parser.add_argument('--decay_step',  
                    type=int,
                    default=6000)
#every decay_step,lr = lr*decay_rate
parser.add_argument('--decay_rate',type=float,  
                    default=0.1)
#the number of epochs to train
parser.add_argument('--epochs', 
                    type=int,
                    default=120) 
#which GPU to use
parser.add_argument('--gpu',
                    type=str,
                    default=0)
flags = parser.parse_args()  

def set_config():
    """
    This function set the GUP option
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu)  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)  
    config = tf.ConfigProto(gpu_options=gpu_options)  
    session = tf.Session(config=config)

def data_augmentation(image, label, training=True):
    """
    This function expends data with a random flip way
    Args:
    image: input image with three channels
    label: the corresponding ground truth of the input image with one channel
    Returns:
    image,label
    """
    if training:
        image_label = tf.concat([image, label], axis=-1)  
        print('image label shape concat', image_label.get_shape()) 
        maybe_flipped = tf.image.random_flip_left_right(image_label)  
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)         
        image = maybe_flipped[:, :, :-1] 
        mask = maybe_flipped[:, :, -1:]  
        return image, mask

def read_csv(queue, augmentation=True):
    """
    This function reads images and corresponding ground truth from .csv file
    """
    csv_reader = tf.TextLineReader(skip_header_lines=0)  
    _, csv_content = csv_reader.read(queue) 
    image_path, label_path = tf.decode_csv(csv_content, record_defaults=[[""], [
        ""]]) 
    image_file = tf.read_file(image_path)  
    label_file = tf.read_file(label_path)  
    image = tf.image.decode_jpeg(image_file, channels=3)  
    image.set_shape([flags.h, flags.w, flags.c_image])  
    image = tf.cast(image, tf.float32)  
    print('image shape', image.get_shape())  
    label = tf.image.decode_png(label_file, channels=1)  
    label.set_shape([flags.h, flags.w, flags.c_label])  
    label = tf.cast(label, tf.float32)
    #normalize the label to[0,1]
    label = label / (tf.reduce_max(label))  
    print('label shape', label.get_shape())  

    if augmentation:
        image, label = data_augmentation(image, label) 
    else:
        pass  
    return image, label


def main(flags):  
    
    current_time = time.strftime("%m/%d/%H/%M/%S")  
    train_logdir = os.path.join(flags.logdir, "train", current_time)  
    validation_logdir = os.path.join(flags.logdir, "validation", current_time)  #

    train = pd.read_csv(flags.training_dir)  
    num_train = train.shape[0]  
    
    validation = pd.read_csv(flags.validation_dir)  
    num_validation = validation.shape[0] 

    tf.reset_default_graph() 
    #set the placeholder for image and ground truth
    X = tf.placeholder(tf.float32, shape=[flags.batch_size, flags.h,flags.w, flags.c_image], name='X') 
    y = tf.placeholder(tf.float32, shape=[flags.batch_size, flags.h, flags.w, flags.c_label], name='y')
    #set the placeholder for training mode
    training = tf.placeholder(tf.bool, name='training')  
    #get the output of the network
    score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse = model.unet(
        X,flags.batch_size,flags.h,flags.w, training=True)  
    print(upscore_fuse.get_shape().as_list()) 

    #the cross_entropy loss 
    loss5 = loss_CE(score_dsn5_up, y)
    loss4 = loss_CE(score_dsn4_up, y)
    loss3 = loss_CE(score_dsn3_up, y)
    loss2 = loss_CE(score_dsn2_up, y)
    loss1 = loss_CE(score_dsn1_up, y)
    loss_fuse = loss_CE(upscore_fuse, y)
    #add all of the output to tensorboard scalar for  visualization
    tf.summary.scalar("CE5", loss5)
    tf.summary.scalar("CE4", loss4)
    tf.summary.scalar("CE3", loss3)
    tf.summary.scalar("CE2", loss2)
    tf.summary.scalar("CE1", loss1)
    tf.summary.scalar("CE_fuse", loss_fuse)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')  

    #sets the decay rate of aaf loss
    dec = tf.pow(10.0, (tf.cast(-(global_step/int(num_train/flags.batch_size*flags.epochs)), tf.float32)))

    w_edge = tf.get_variable(
        name='edge_w',
        shape=(1, 1, 1, 2, 1, 3),  
        dtype=tf.float32,
        initializer=tf.constant_initializer(0))  
    w_edge = tf.nn.softmax(w_edge, dim=-1)  
    w_not_edge = tf.get_variable(
        name='nonedge_w',
        shape=(1, 1, 1, 2, 1, 3),
        dtype=tf.float32,
        initializer=tf.constant_initializer(0))  
    w_not_edge = tf.nn.softmax(w_not_edge, dim=-1)  
   

    score_dsn5_up = tf.nn.sigmoid(score_dsn5_up)
    score_dsn4_up = tf.nn.sigmoid(score_dsn4_up)
    score_dsn3_up = tf.nn.sigmoid(score_dsn3_up)
    score_dsn2_up = tf.nn.sigmoid(score_dsn2_up)
    score_dsn1_up = tf.nn.sigmoid(score_dsn1_up)
    upscore_fuse = tf.nn.sigmoid(upscore_fuse,name='output')
    
    upscore_fuse_0 = 1 - upscore_fuse
    prob = tf.concat([upscore_fuse_0, upscore_fuse], axis=-1)
    # aaf_loss = loss_aaf(y, upscore_fuse)

    labels = tf.cast(y, tf.uint8)
    one_hot_lab = tf.one_hot(tf.squeeze(labels, axis=-1), depth=2)  
    aaf_losses = []
    eloss_1, neloss_1 = lossx.adaptive_affinity_loss(labels,
                                                     one_hot_lab,
                                                     prob,
                                                     1,
                                                     2,
                                                     3,
                                                     w_edge[..., 0],  
                                                     w_not_edge[..., 0])
    # Apply AAF on 5x5 patch.
    eloss_2, neloss_2 = lossx.adaptive_affinity_loss(labels,
                                                     one_hot_lab,
                                                     prob,
                                                     2,
                                                     2,
                                                     3,
                                                     w_edge[..., 1],
                                                     w_not_edge[..., 1])
    # Apply AAF on 7x7 patch.
    eloss_3, neloss_3 = lossx.adaptive_affinity_loss(labels,
                                                     one_hot_lab,
                                                     prob,
                                                     3,
                                                     2,
                                                     3,
                                                     w_edge[..., 2],
                                                     w_not_edge[..., 2])
    #decays aaf loss with the increase of global step
    aaf_loss = tf.reduce_mean(eloss_1) * dec
    aaf_loss += tf.reduce_mean(eloss_2) * dec
    aaf_loss += tf.reduce_mean(eloss_3) * dec
    aaf_loss += tf.reduce_mean(neloss_1) * dec
    aaf_loss += tf.reduce_mean(neloss_2) * dec
    aaf_loss += tf.reduce_mean(neloss_3) * dec
    aaf_losses.append(aaf_loss)  
    

    # Sum all loss terms.
    mean_seg_loss = loss5 + loss4 + loss3 + loss2 + loss1 + loss_fuse
    mean_aaf_loss = tf.add_n(aaf_losses)
    CE_total = mean_seg_loss + mean_aaf_loss
    
    tf.summary.scalar("CE_total", CE_total)
    tf.summary.scalar("mean_seg_loss", mean_seg_loss)
    tf.summary.scalar("mean_aaf_loss", mean_aaf_loss)
    tf.summary.scalar("dec", dec)

    # Grab variable names which are used for training
    all_trainable = tf.trainable_variables()
    fc_trainable = [v for v in all_trainable
                      if 'block' not in v.name and 'edge' not in v.name] # lr*1
    base_trainable = [v for v in all_trainable if 'block' in v.name] # lr*10
    aaf_trainable = [v for v in all_trainable if 'edge' in v.name]

    # Computes gradients per iteration.
    grads = tf.gradients(CE_total, base_trainable+fc_trainable+aaf_trainable)
    grads_base = grads[0:len(base_trainable)]
    grads_fc = grads[len(base_trainable):len(base_trainable)+len(fc_trainable)]
    grads_aaf = grads[len(base_trainable)+len(fc_trainable):]
    grads_aaf = [-g for g in grads_aaf]


    
    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, 
                                               decay_steps=flags.decay_step,  
                                               decay_rate=flags.decay_rate,
                                               staircase=True)  
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  
    with tf.control_dependencies(update_ops):  
        opt_base = tf.train.AdamOptimizer(10*learning_rate)
        opt_fc = tf.train.AdamOptimizer(learning_rate)
        opt_aaf = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.train.get_or_create_global_step() 
 
        # Define tensorflow operations which apply gradients to update variables.
        train_op_base = opt_base.apply_gradients(zip(grads_base, base_trainable))
        train_op_fc = opt_fc.apply_gradients(zip(grads_fc, fc_trainable))
        train_op_aaf = opt_aaf.apply_gradients(zip(grads_aaf, aaf_trainable),global_step=global_step)
        train_op = tf.group(train_op_base, train_op_fc, train_op_aaf)
        
    train_csv = tf.train.string_input_producer(['train.csv'])  
    validation_csv = tf.train.string_input_producer(['validation.csv']) 
    #get the training and validation data
    train_image, train_label = read_csv(train_csv, augmentation=True)  
    validation_image, validation_label = read_csv(validation_csv, augmentation=False)  
    
    X_train_batch_op, y_train_batch_op = tf.train.shuffle_batch([train_image, train_label], batch_size=flags.batch_size,                                    
                                                                capacity=flags.batch_size * 500,
                                                                min_after_dequeue=flags.batch_size * 100,
                                                                allow_smaller_final_batch=True)  

    X_validation_batch_op, y_validation_batch_op = tf.train.batch([validation_image, validation_label], batch_size=flags.batch_size,
                                                      capacity=flags.batch_size * 20, allow_smaller_final_batch=True)

    print('Shuffle batch done')  
    #add all of the output into collection
    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', training)
    tf.add_to_collection('score_dsn5_up', score_dsn5_up)
    tf.add_to_collection('score_dsn4_up', score_dsn4_up)
    tf.add_to_collection('score_dsn3_up', score_dsn3_up)
    tf.add_to_collection('score_dsn2_up', score_dsn2_up)
    tf.add_to_collection('score_dsn1_up', score_dsn1_up)
    tf.add_to_collection('upscore_fuse', upscore_fuse)

    #add all of the output to tensorboard image for visualization
    tf.summary.image('Input Image:', X)
    tf.summary.image('Label:', y)
    tf.summary.image('score_dsn5_up:', score_dsn5_up)
    tf.summary.image('score_dsn4_up:', score_dsn4_up)
    tf.summary.image('score_dsn3_up:', score_dsn3_up)
    tf.summary.image('score_dsn2_up:', score_dsn2_up)
    tf.summary.image('score_dsn1_up:', score_dsn1_up)
    tf.summary.image('upscore_fuse:', upscore_fuse)
    
    #add the learning rate into tensorboard scalar for visualization
    tf.summary.scalar("learning_rate", learning_rate)
    
    #add all of the output to tensorboard histogram for visualization
    tf.summary.histogram('score_dsn1_up:', score_dsn1_up)
    tf.summary.histogram('score_dsn2_up:', score_dsn2_up)
    tf.summary.histogram('score_dsn3_up:', score_dsn3_up)
    tf.summary.histogram('score_dsn4_up:', score_dsn4_up)
    tf.summary.histogram('score_dsn5_up:', score_dsn5_up)
    tf.summary.histogram('upscore_fuse:', upscore_fuse)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)  
        validation_writer = tf.summary.FileWriter(
            validation_logdir)  
        init = tf.global_variables_initializer()  
        sess.run(init) 
        saver = tf.train.Saver() 
        try:
            coord = tf.train.Coordinator()  
            threads = tf.train.start_queue_runners(coord=coord) 
            for epoch in range(flags.epochs): 
                #feed the network with training data
                for step in range(0, num_train, flags.batch_size):  
                    X_train, y_train = sess.run([X_train_batch_op, y_train_batch_op])  
                    _, step_ce, step_summary, global_step_value = sess.run([train_op, CE_total, summary_op, global_step],
                                                                           feed_dict={X: X_train, y: y_train,  
                                                                                      training: True})                    

                    train_writer.add_summary(step_summary, global_step_value)  
                    print(
                    'epoch:{} step:{} loss_CE:{}'.format(epoch + 1, global_step_value, step_ce)) 
                #feed the network with validation data   
                for step in range(0, num_validation, flags.batch_size):
                    X_test, y_test = sess.run([X_validation_batch_op, y_validation_batch_op])  
                    step_ce, step_summary = sess.run([CE_total, summary_op], feed_dict={X: X_test, y: y_test,
                                                                                    training: False}) 

                    validation_writer.add_summary(step_summary, epoch * (
                        num_train // flags.batch_size) + step // flags.batch_size * num_train // num_validation)  
                    print('validation loss_CE:{}'.format(step_ce))  
                saver.save(sess, '{}/model.ckpt'.format(flags.model_dir))  
        finally:
            coord.request_stop()  
            coord.join(threads)  
            saver.save(sess, "{}/model.ckpt".format(flags.model_dir)) 


if __name__ == '__main__':
    set_config()  
    main(flags) 
