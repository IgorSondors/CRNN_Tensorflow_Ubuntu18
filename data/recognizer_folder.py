#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use shadow net to recognize the scene text of a single image
"""
import time
start_time = time.time()

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
import shutil

sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/config')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/local_utils')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data_provider')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/crnn_model')
sys.path.append('/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18')

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg

#######################################
'''import server
dir(server)'''
#import Detector_script_clean_orig
import Detector_script_clean
import dict_correction
import vis




def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_list', type=str,
                        help='Path to the image to be tested', default = image_list )
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use', default = one_of_three_models)
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored', default = char_dic)
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored', default = ord_map)

    return parser.parse_args()


def input_rec(image_list, weights_path, char_dict_path, ord_map_dict_path):
    new_heigth = 32

    new_width = CFG.ARCH.INPUT_SIZE[0]

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
    return inputdata, codec, net, inference_ret, decodes, _

def recognize(image_list, weights_path, char_dict_path, ord_map_dict_path):
    """

    :param image_list:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :return:
    """
    Recognition_result = []
    
    inputdata, codec, net, inference_ret, decodes, _ = input_rec(image_list, weights_path, char_dict_path, ord_map_dict_path)
    # config tf saver
    saver = tf.compat.v1.train.Saver()

    # config tf session
    sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.compat.v1.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)
        for i in range(len(image_list)):
               
            preds, confidence = sess.run([decodes, _], feed_dict={inputdata: [image_list[i]]})
            confidence_list.append(confidence[0][0])
            

            preds = codec.sparse_tensor_to_str(preds[0])[0]

            #print('crop №', i, ':', (preds, confidence[0][0]))
        
            Recognition_result.append(preds)
            #print(Recognition_result)

    sess.close()

    return Recognition_result


crop_folder = '/home/sondors/Recognizer/server/not_api/server_crops'



one_of_three_models = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/model/crnn_syn90k (copy)/shadownet_2020-05-22-02-14-55.ckpt-34000'
char_dic = '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/char_dict.json'
ord_map =  '/home/sondors/Recognizer/CRNN_Tensorflow_Ubuntu18/data/char_dict/ord_map.json'






#PATH_TO_IMAGE = '/home/sondors/Recognizer/server/not_api/img/photo_2020-06-02_14-48-12.jpg'

folder_path = '/home/sondors/Recognizer/server/not_api/images'
sys.path.append(folder_path)
for PATH_TO_IMAGE in next(os.walk(folder_path))[2]:
    try:
        P_T_I = PATH_TO_IMAGE
        #print(PATH_TO_IMAGE)
        PATH_TO_IMAGE = os.path.join(folder_path, '{}'.format(PATH_TO_IMAGE))
        img = cv2.imread(PATH_TO_IMAGE)
        #print('Shape is', np.shape(img))
        image_list =[]
        confidence_list = []

        Detector_script_clean.Writing_i_k_Crops(img)
        res, img_shape, probability_list = Detector_script_clean.Writing_i_k_Crops(img)

        quantity_of_files = len(next(os.walk(crop_folder))[2])
        print('Recognizer received res, img_shape, probability_list: ',res, img_shape, probability_list)


        for i in range(quantity_of_files):



            #load and normalize an image

            one_of_images =  os.path.join(crop_folder, '{}.jpg'.format(i))
            image =cv2.imread(one_of_images, cv2.IMREAD_COLOR)
            new_heigth = 32

            new_width = CFG.ARCH.INPUT_SIZE[0]
            
            image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)
            image_list.append(np.array(image, np.float32) / 127.5 - 1.0)

            os.remove(one_of_images)#Remove crops from crop_folder



        args = init_args()



        input_rec(
            image_list=args.image_list,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path,
        )


        text_output = recognize(
            image_list=args.image_list,
            weights_path=args.weights_path,
            char_dict_path=args.char_dict_path,
            ord_map_dict_path=args.ord_map_dict_path,
        )

        #print('text_output =',text_output)
        #print('res =', res, len(res))
        #print('img_shape = ', img_shape)
        transfer = []
        dict_correction.Strings_to_dict(text_output, res,  confidence_list,  img_shape, transfer)

        text_output_dict = transfer[0]
        res = transfer[1]
        #print(text_output_dict)
        #print(res)
        
        l = open(os.path.join('/home/sondors/Recognizer/server/not_api/recogn', '{}.txt'.format(P_T_I)),'w', encoding="utf8")
        keys = ['Паспорт выдан', 'Дата выдачи', 'Код подразделения', 'Фамилия', 'Имя', 'Отчество', 'Пол', 'Дата рождения', 'Место рождения']
        for i in keys:
            text = text_output_dict[i]
            l.write(text+'\n')

        visualize = False
        if visualize:
            vis.Vis_recogn(PATH_TO_IMAGE, text_output_dict, res, img_shape)

    except:

        folder_from = folder_path
        names = '/home/sondors/Recognizer/server/not_api/recogn'
        folder_to = '/home/sondors/Recognizer/server/not_api/img (copy)/later'


        shutil.copyfile(os.path.join(folder_from, '{}'.format(P_T_I)), os.path.join(folder_to, '{}'.format(P_T_I)))

        os.remove(os.path.join(folder_from, '{}'.format(P_T_I)))
        print('Recognizer exeption processed, bad image ', P_T_I, ' is removed')



time_spent = (time.time() - start_time)
print('time_spent:', time_spent)