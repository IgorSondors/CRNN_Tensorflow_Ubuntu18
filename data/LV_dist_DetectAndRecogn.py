# -*- coding: utf-8-sig -*-
import Levenshtein as lv
import glob
import os
import re
import codecs

names_path = r'C:\Users\sondors\Desktop\400pass\parsed'
names = next(os.walk(names_path))[2]

dist_between_allPasp = []

char_true_counter = 0
char_recogn_counter = 0
dist_ALLpassp_counter = 0

for j in names:

    dist_in_1passp = []
    dist_1passp_counter = 0
    char_1passp_counter = 0
    
    true = codecs.open(os.path.join(r'C:\Users\sondors\Desktop\400pass\parsed_changed', '{}'.format(j)),'r', encoding="utf8")
    recogn = codecs.open(os.path.join(r'C:\Users\sondors\Desktop\400pass\img_rec_txt', '{}'.format(j)),'r', encoding="utf8")
    

    recogn_ = recogn.readlines()
    true_ = true.readlines()

    crop_recogn = list(recogn_)
    crop_true = list(true_)
    #print(j)
    #print(crop_true)
    #Убираем символы переноса каретки
    for i in range(len(crop_recogn)):
        #print(len(crop_recogn))
        crop_recogn[i] = re.sub('[\r\n\n]', '', crop_recogn[i])
        crop_true[i] = re.sub('[\r\n\n]', '', crop_true[i])
        dist_in_1passp.append(lv.distance(crop_recogn[i], crop_true[i]))

        char_1passp_counter = char_1passp_counter + len(crop_recogn[i])
        dist_1passp_counter = dist_1passp_counter + dist_in_1passp[i]

        dist_ALLpassp_counter = dist_ALLpassp_counter + dist_in_1passp[i]
        char_true_counter = char_true_counter + len(crop_true[i])
        char_recogn_counter = char_recogn_counter + len(crop_recogn[i])





    dist_between_allPasp.append(dist_1passp_counter)
    for i in dist_between_allPasp:
        with open('dist.txt','a', encoding="utf8") as d:
            d.write(str(i) + '      ,' + 'in file ' + j + '\n')

print('char_true_counter:', char_true_counter)
print('char_recogn_counter:', char_recogn_counter)
print('dist_ALLpassp_counter:', dist_ALLpassp_counter)
print('Accuracy is ', 100-(dist_ALLpassp_counter/char_true_counter*100))
