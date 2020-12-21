import glob
import os
import re
import csv



pattern = r"(\d.)+"

#re.sub('[ ]', '', region_attributes)
for name in next(os.walk(r'C:\Users\sondors\Desktop\400pass\parsed'))[2]:
    with open(os.path.join(r'C:\Users\sondors\Desktop\400pass\parsed', '{}'.format(name)),'r', encoding="utf8") as l:
        list_of_strings = []
        for i in l:
            list_of_strings.append(i[:-1])
        print(list_of_strings)
        i = 1
        while not re.match(pattern, list_of_strings[i]):
            
            list_of_strings[0] = list_of_strings[0] + ' ' + list_of_strings[i]
            list_of_strings.remove(list_of_strings[i])
        print(list_of_strings)

        i  = -1
        check_next = True
        while check_next:

            if not re.match(pattern, list_of_strings[i]):
                i = i-1

            else:
                check_next = False
                list_of_strings_trail = list_of_strings[i+1:]
                trail = ['']
                for j in list_of_strings_trail:
                    trail[0] = trail[0] + ' ' + j
                
                list_of_strings = list_of_strings[:i+1]
                
                list_of_strings.append(trail[0])
        
        with open(os.path.join(r'C:\Users\sondors\Desktop\400pass\parsed_changed', '{}'.format(name)),'a', encoding="utf8") as m:
            for i in list_of_strings:
                m.write(i + '\n')

        print(list_of_strings)



