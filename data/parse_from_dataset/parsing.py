import glob
import os
import re
import csv


keys = ['Паспортвыдан', 'Датавыдачи', 'Кодподразделения', 'Фамилия', 'Имя', 'Отчество', 'Пол', 'Датарождения', 'Месторождения', 'РОССИЙСКАЯФЕДЕРАЦИЯ', 'Личнаяподпись', 'Личныйкод']

file2reader = csv.reader(open('train.txt'), delimiter=",")

names = r'C:\Users\sondors\Desktop\400pass\recognized'


for filename, region_attributes in file2reader:

    if filename in next(os.walk(names))[2] and re.sub('[ ]', '', region_attributes) not in keys:
        with open(os.path.join(r'C:\Users\sondors\Desktop\400pass\parsed', '{}.txt'.format(filename)),'a', encoding="utf8") as l:
            l.write(region_attributes + '\n')

        