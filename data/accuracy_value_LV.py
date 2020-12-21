# -*- coding: utf-8-sig -*-
import Levenshtein as lv
import glob
import os
import re
import codecs

recogn = codecs.open( r"C:\Users\Igor\Desktop\crop_recogn_12000.txt", "r", "utf-8" )
true = codecs.open( r"C:\Users\Igor\Desktop\crop_true_12000.txt", "r", "utf-8" )

recogn_ = recogn.readlines()
true_ = true.readlines()

crop_recogn = list(recogn_)
crop_true = list(true_)

#Убираем символы переноса каретки
for i in range(len(crop_recogn)):
    crop_recogn[i] = re.sub('[\r\n\n]', '', crop_recogn[i])
    crop_true[i] = re.sub('[\r\n\n]', '', crop_true[i])


dist = []

char_counter = 0
char_counter_recogn = 0

for i in range(12000):
    dist.append((lv.distance(crop_recogn[i], crop_true[i])))#/len(crop_true[i]))
    char_counter = char_counter + len(crop_true[i])
    char_counter_recogn = char_counter_recogn + len(crop_recogn[i])


train = open(r"C:\Users\Igor\Desktop\dist.txt","w", encoding="utf8")
for i in dist:
    train.write(str(i)+"\n")

summ_of_dist = 0


print('Количество строк рассмотрено:', len(dist))

for i in dist:
    summ_of_dist = summ_of_dist + i

print('Количество ошибок по Левенштайну', summ_of_dist)

print('Кол-во символов в разметке:', char_counter)
print('Кол-во символов распознано:', char_counter_recogn)
print('Доля ошибок', summ_of_dist/char_counter)
print('Процент ошибок', summ_of_dist/char_counter*100)



"""print(len(dist))

print((crop_true[:2]))
print(len(crop_true[0]))

print((crop_recogn[:2]))
print(len(crop_recogn[0]))

print((lv.distance(crop_recogn[0], crop_true[0])))"""