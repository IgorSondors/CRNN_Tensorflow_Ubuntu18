#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import figure
import matplotlib
from PIL import Image
import numpy as np
import cv2
import os
os.environ['DISPLAY'] = ':0'

def Vis_recogn(PATH_TO_IMAGE, text_output_dict, res, img_shape):

    keys = ['Паспорт выдан', 'Дата выдачи', 'Код подразделения', 'Фамилия', 'Имя', 'Отчество', 'Пол', 'Дата рождения', 'Место рождения']
    
    im = np.array(Image.open(PATH_TO_IMAGE), dtype=np.uint8)

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    for i in range(len(keys)):
        text = text_output_dict[keys[i]]
        x0 = res[keys[i]][0][0]
        y0 = res[keys[i]][0][1]
        x1 = res[keys[i]][0][2]
        y1 = res[keys[i]][0][3]
        W = x1-x0
        H = y1-y0
        w = (x0+x1) / 2
        h = (y0+y1) / 2
        position = (w, h)
        rect = matplotlib.patches.Rectangle((x0,y0), W, H,linewidth=1,edgecolor='r',facecolor='none')
        matplotlib.pyplot.text(x0,y0, text, fontdict=None)
        ax.add_patch(rect)

    #plt.show()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('testplot.png', dpi=100)
    cv2.imshow('qwert', cv2.imread('testplot.png'))
    cv2.waitKey()   
    cv2.destroyAllWindows() 

"""PATH_TO_IMAGE = '/home/sonders/Recognizer/server/not_api/photo_2020-06-02_14-53-03.jpg'
text_output_dict = {'Паспорт выдан': 'ТП № 65 ОУФМС РОССИИ ПО САНКТ-ПЕТЕРБУРГУ И ЛЕНИНГРАДСКОЙ ОБЛ. В ПРИМОРСКОМ РАЙОНЕ Г. САНКТ-ПЕТЕРБУРГА', 'Дата выдачи': '12.04.2016', 'Код подразделения': '780-065', 'Фамилия': 'БОРОДАТОВ', 'Имя': 'АНтеН', 'Отчество': 'МИХАЙЛОВИЧ', 'Пол': 'МУЖ.', 'Дата рождения': '12.05.1981', 'Место рождения': ' ГОР. ЛЕНИНГРАД'}
res = {'Паспорт выдан': [[201, 176, 797, 223], [85, 224, 942, 276], [140, 275, 866, 320]], 'Дата выдачи': [[192, 332, 401, 371]], 'Код подразделения': [[570, 327, 729, 369]], 'Фамилия': [[537, 752, 732, 792]], 'Имя': [[572, 839, 698, 876]], 'Отчество': [[522, 881, 757, 923]], 'Пол': [[390, 926, 488, 968]], 'Дата рождения': [[582, 920, 806, 971]], 'Место рождения': [[479, 975, 818, 1024]]}
img_shape = 0
Vis_recogn(PATH_TO_IMAGE, text_output_dict, res, img_shape)"""