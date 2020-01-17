# coding:utf-8
import os
import cv2    
import glob   
import tensorflow as tf
import numpy as np
import argparse    
import time
ep=1e-8
batch_size = 5  
parser = argparse.ArgumentParser()
# you can optionly change the input_dir/model_dir/save_dir params
parser.add_argument('--input_dir',
                    type=str,
                    default=r'./dataset/test')
parser.add_argument('--model_dir',
                    type=str,
                    default=r'./model1')
parser.add_argument('--save_dir',
                    type=str,
                    default=r'./result1')
flags = parser.parse_args()
 
def load_model():
    # restore the model and set the GPU options
    file_meta = os.path.join(flags.model_dir, 'model.ckpt.meta')   
    file_ckpt = os.path.join(flags.model_dir, 'model.ckpt')   
    saver = tf.train.import_meta_graph(file_meta)    
    sess = tf.InteractiveSession() 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)  
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver.restore(sess, file_ckpt)    
    return sess

def read_image(image_path, gray=False):
    # read the test image and change the channels from BGR to RGB
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
    else:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   

def main(flags):
    sess = load_model()    
    X = tf.get_collection('inputs')[0]   
    training = tf.get_collection('inputs')[1] 
    pred = tf.get_collection('upscore_fuse')[0]   
    names=os.listdir(flags.input_dir)   
    num_images = 0     
    num_batchs = 0   
    image_feed = []
    meantime=[]      
    for name in names:
        inputname=os.path.join(flags.input_dir,name)   
        image = read_image(inputname)   
        num_images += 1                 
        image_feed.append(image)
        if num_images == batch_size:
            starttime=time.time()
            label_pred = sess.run(pred, feed_dict={X: image_feed, training: False})   
            endtime=time.time()
            print('test time per batch:'+str(endtime-starttime)+'s')
            meantime.append(endtime-starttime)
                   
            for i in range(label_pred.shape[0]):
                merged = np.squeeze(label_pred[i]) 
                #normoralize the input to[-1,1]   
                merged=np.uint8((merged)*255)
                crt_name = names[num_batchs*batch_size+i]    
                save_name = os.path.join(flags.save_dir, crt_name[:-4]+".png")   
                cv2.imwrite(save_name, merged)   
                print('Pred saved')           
            image_feed = []   
            num_images = 0   
            num_batchs += 1
    print('mean test time per batch:'+str(np.mean(meantime[2:]))+'s') 
  
if __name__ == '__main__':  
      
    main(flags)   

