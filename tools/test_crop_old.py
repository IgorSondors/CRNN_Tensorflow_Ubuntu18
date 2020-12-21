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


def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested', default = one_of_images)
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use', default = one_of_three_models)
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored', default = char_dic)
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored', default = ord_map)
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=True,
                        help='Whether to display images')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def recognize(image_path, weights_path, char_dict_path, ord_map_dict_path, is_vis, is_english=False):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_vis:
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    new_heigth = 32
    scale_rate = new_heigth / image.shape[0]
    new_width = int(scale_rate * image.shape[1])
    new_width = new_width if new_width > CFG.ARCH.INPUT_SIZE[0] else CFG.ARCH.INPUT_SIZE[0]
    print('image', i, 'new width', new_width)
    image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = np.array(image, np.float32) / 127.5 - 1.0

    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, new_heigth, new_width, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

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
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        preds = sess.run(decodes, feed_dict={inputdata: [image]})

        preds = codec.sparse_tensor_to_str(preds[0])[0]
        Recognition_result.append(preds)#Добавлено мною
        
        if is_english:
            preds = ' '.join(wordninja.split(preds))

        logger.info('Predict image {:s} result: {:s}'.format(
            ops.split(image_path)[1], preds)
        )

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.show()

    sess.close()

    return


Recognition_result = []
quantity_of_files = len(next(os.walk('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/ubunt_for/crop_old'))[2])

#######


for i in range(quantity_of_files): #in range(len(res)) если брать кропы из детектора
    one_of_images =  '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/ubunt_for/crop_old/{}.jpg'.format(i)
    one_of_images_shape = cv2.imread(one_of_images)

    one_of_images_width = np.shape(one_of_images_shape)[:2][1]
    print('Detection of', i, 'image')
    print(i, 'image width', one_of_images_width)

    ####test_img = open("/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/ubunt_for/crop_names_test.txt",'w', encoding="utf8")
   
    #one_of_three_models = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k_after22.04/shadownet_2020-04-24-01-00-37.ckpt-12000'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k/shadownet_2020-04-27-22-41-59.ckpt-16000'
    #char_dic = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict_after22.04/char_dict.json'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/char_dict.json'
    #ord_map =  '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict_after22.04/ord_map.json'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/ord_map.json'
    
    one_of_three_models = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k (copy)/shadownet_2020-05-22-02-14-55.ckpt-34000'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k/shadownet_2020-05-21-15-10-49.ckpt-28000'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k (another copy)/shadownet_2020-04-30-23-25-20.ckpt-44000'
    char_dic = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/char_dict.json'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict_01.05/char_dict.json'
    ord_map =  '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/ord_map.json'#'/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict_01.05/ord_map.json'
 

    if __name__ == '__main__':
        """
        
        """
        # init images
        args = init_args()

        # detect images
        recognize(
            image_path=args.image_path,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path,
            is_vis=args.visualize
        )

print('Recognition result is ', Recognition_result)

with open("/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/ubunt_for/crop_names_test.txt",'w', encoding="utf8") as test_img:
    for j in range(len(Recognition_result)):
        test_img.write(Recognition_result[j]+ '\n')

print(Recognition_result)

print("--- %s seconds ---" % (time.time() - start_time))