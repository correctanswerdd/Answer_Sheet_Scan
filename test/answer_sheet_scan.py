import json
import cv2
import os
import numpy as np
from ImportGraph import ImportGraph
# import sys
#
# sys.path.append('network_cfg/')

user_dir = "data/2/"

with open(user_dir + "template.json", 'r') as load_f:
    load_template = json.load(load_f)
    sub_list = load_template[0]["choice"]
    opt_list = []
    for i in sub_list:
        opt_list.append(i["options"])



def check_opt(pdt):
    p = np.argmax(pdt, 1)
    if p == 0:
        return 'A'
    elif p == 1:
        return 'B'
    elif p == 2:
        return 'C'
    elif p == 3:
        return 'D'
    elif p == 4:
        return 'E'
    elif p == 5:
        return 'F'
    elif p == 6:
        return 'G'
    else:
        pass


for root, dirs, files in os.walk(user_dir, topdown=False):
    opt7 = ImportGraph("network_wb/my_net_7/save_net.ckpt", 7)
    opt4 = ImportGraph("network_wb/my_net_4/save_net.ckpt", 4)
    opt3 = ImportGraph("network_wb/my_net_3/save_net.ckpt", 3)
    for i in files:
        if i == '.DS_Store':
            continue
        if i == 'template.json':
            continue
        img = cv2.imread(root + "/" + i)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        n = 0
        for opt in opt_list:
            n += 1
            print(n)
            if len(opt) == 3:
                x, y = opt[0][0], opt[0][1]
                crop = img[opt[0][1]:opt[2][3]][:, opt[0][0]:opt[2][2]]
                crop = cv2.resize(crop, (110, 14), interpolation=cv2.INTER_CUBIC)
                x_img = crop[np.newaxis, :]
                predict = opt3.predict(x_img)
                cv2.putText(img, check_opt(predict), (opt[2][2], opt[2][3]), font, 1.2, (0, 0, 0), 1)
                # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
                cv2.rectangle(img, pt1=(x, y), pt2=(opt[2][2], opt[2][3]), color=(0, 0, 255), thickness=3)
                pass
            elif len(opt) == 4:
                x, y = opt[0][0], opt[0][1]
                crop = img[opt[0][1]:opt[3][3]][:, opt[0][0]:opt[3][2]]
                crop = cv2.resize(crop, (152, 16), interpolation=cv2.INTER_CUBIC)
                x_img = crop[np.newaxis, :]
                predict = opt4.predict(x_img)
                cv2.putText(img, check_opt(predict), (opt[3][2], opt[3][3]), font, 1.2, (0, 0, 0), 1)
                # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
                cv2.rectangle(img, pt1=(x, y), pt2=(opt[3][2], opt[3][3]), color=(0, 0, 255), thickness=3)
            elif len(opt) == 7:
                x, y = opt[0][0], opt[0][1]
                crop = img[opt[0][1]:opt[6][3]][:, opt[0][0]:opt[6][2]]
                crop = cv2.resize(crop, (232, 16), interpolation=cv2.INTER_CUBIC)
                x_img = crop[np.newaxis, :]
                predict = opt7.predict(x_img)
                cv2.putText(img, check_opt(predict), (opt[6][2], opt[6][3]), font, 1.2, (0, 0, 0), 1)
                # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
                cv2.rectangle(img, pt1=(x, y), pt2=(opt[6][2], opt[6][3]), color=(0, 0, 255), thickness=3)
            else:
                pass
        cv2.imwrite("output.jpg", img)