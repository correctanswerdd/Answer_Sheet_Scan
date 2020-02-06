import os

img_dir = "data/English/result_c"

for root, dirs, files in os.walk(img_dir, topdown=False):
    for i in files:
        if i[12] == '2':
            os.remove(root + "/" + i)
