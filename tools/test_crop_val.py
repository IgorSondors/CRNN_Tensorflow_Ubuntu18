import time
start_time = time.time()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use shadow net to recognize the scene text of a single image
"""

import argparse
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glog as logger
import wordninja
import sys 
import os

sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/config')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/local_utils')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data_provider')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/crnn_model')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18')

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg

import codecs
import re

def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested')#, default = one_of_images)
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use', default = weights_path)
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored', default = char_dic)
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored', default = ord_map)

    return parser.parse_args()



def recognize(image_path, weights_path, char_dict_path, ord_map_dict_path):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :return:
    """
    image_list =[]
    new_width_list = []
    inputdata_list = []
 
    for i in range(len(test_img)):#(quantity_of_files):
        print(i, 'image processing')

        #load and normalize an image
        one_of_images =  test_img[i]
        image =cv2.imread(one_of_images, cv2.IMREAD_COLOR)
        new_heigth = 32
        scale_rate = new_heigth / image.shape[0]
        new_width = int(scale_rate * image.shape[1])
    
        new_width = CFG.ARCH.INPUT_SIZE[0]#new_width if new_width > CFG.ARCH.INPUT_SIZE[0] else CFG.ARCH.INPUT_SIZE[0]
        new_width_list.append(new_width)
       
        image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)
        image_list.append(np.array(image, np.float32) / 127.5 - 1.0)

    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, new_heigth, new_width, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )
    #inputdata_list.append(inputdata)

    codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=tf.AUTO_REUSE #было#False
    )

    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=int(new_width / 4) * np.ones(1),
        merge_repeated=False,
        beam_width=10
    )
    
    # config tf saver
    saver = tf.compat.v1.train.Saver()

    # config tf session
    sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.compat.v1.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        for i in range(len(test_img)):#(quantity_of_files):
        
            #print('size of image', i+500, 'is',sys.getsizeof(image_list[i]))
            preds = sess.run(decodes, feed_dict={inputdata: [image_list[i]]})
            #preds = sess.run(decodes, feed_dict={inputdata_list[i]: [image_list[i]]})

            preds = codec.sparse_tensor_to_str(preds[0])[0]
            #print(i, 'image recognition result_txt is', preds)
            #print('size of preds', i+500, 'is',sys.getsizeof(preds))
            result_txt.write(preds+ '\n')
        

    sess.close()
    del image_list
    del inputdata_list
    del new_width_list
    del net
    del saver
    del codec

    del inference_ret
    del decodes
    del inputdata
    return

weights_path = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k (copy)/shadownet_2020-05-22-02-14-55.ckpt-34000'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k/shadownet_2020-05-21-15-10-49.ckpt-28000'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k (another copy)/shadownet_2020-04-30-23-25-20.ckpt-44000'
char_dic = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/char_dict.json'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict_01.05/char_dict.json'
ord_map =  '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/ord_map.json'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict_01.05/ord_map.json'

#quantity_of_files = len(next(os.walk('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/ubunt_for/crop_500'))[2])

#######

with open(("/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/annotations_Over400pix/annotation_val.txt"), "r", encoding="utf8") as f:

    fd = f.readlines()

    #new_fd = list(fd)

    test_img = list(fd)

    for i in range(len(test_img)):
        (test_img[i]) = re.sub('[\r\n\n]', '', (test_img[i]))[:-6]
f.close()



with open("/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/ubunt_for/crop_names_val.txt",'w', encoding="utf8") as result_txt:



    if __name__ == '__main__':
        """
        
        """
        # init images
        args = init_args()



        # recogn images
        recognize(
            image_path=args.image_path,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path,
        )


print("--- %s seconds ---" % (time.time() - start_time))