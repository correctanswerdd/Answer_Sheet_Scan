import cv2
import numpy as np
import json
import math
import os
import pickle
from progressbar import *

template_json_dir = "data/template.json"
userimg_png_dir = "data/English/result_c"


def load_data(json_dir: str):
    with open(json_dir, 'r') as load_f:
        load_template = json.load(load_f)
        sub_list = load_template[0]["choice"]

        opt_list = []
        for i in sub_list:
            opt_list.append(i["options"])
    return opt_list


def get_template_rec(opt_list: list):
    opt3 = 0
    opt3_high = 0.0
    opt3_width = 0.0
    opt4 = 0
    opt4_high = 0.0
    opt4_width = 0.0
    opt7 = 0
    opt7_high = 0.0
    opt7_width = 0.0

    for opt in opt_list:
        if len(opt) == 3:
            opt3_high = (opt3_high * opt3 + opt[2][3] - opt[0][1]) / (opt3 + 1)
            opt3_width = (opt3_width * opt3 + opt[2][2] - opt[0][0]) / (opt3 + 1)
            opt3 += 1
        elif len(opt) == 4:
            opt4_high = (opt4_high * opt4 + opt[3][3] - opt[0][1]) / (opt4 + 1)
            opt4_width = (opt4_width * opt4 + opt[3][2] - opt[0][0]) / (opt4 + 1)
            opt4 += 1
        elif len(opt) == 7:
            opt7_high = (opt7_high * opt7 + opt[5][3] - opt[0][1]) / (opt7 + 1)
            opt7_width = (opt7_width * opt7 + opt[5][2] - opt[0][0]) / (opt7 + 1)
            opt7 += 1
    return (math.ceil(opt3_high), math.ceil(opt3_width), 3), \
           (math.ceil(opt4_high), math.ceil(opt4_width), 3), \
           (math.ceil(opt7_high), math.ceil(opt7_width), 3)


def get_options(opt_list: list, rec3: tuple, rec4: tuple, rec7: tuple, user_dir: str):
    opt3 = []
    opt4 = []
    opt7 = []

    # 遍历路径下所有用户图像
    for root, dirs, files in os.walk(user_dir, topdown=False):
        progress = ProgressBar(len(files) - 1)
        progress.start()
        n = 0
        for i in files:
            img = cv2.imread(root + "/" + i)
            # 对每一张图像，裁剪出所有区域
            for opt in opt_list:
                if len(opt) == 3:
                    crop = np.zeros(rec3)
                    high, width, channel = rec3
                    # crop = img[opt[0][1]:opt[0][1] + high][opt[0][0]:opt[0][0] + width]  numpy提取某几行某几列的错误写法……
                    crop = img[opt[0][1]:opt[0][1] + high][:, opt[0][0]:opt[0][0] + width]
                    # cv2.imshow("image", crop)
                    # cv2.waitKey(0)
                    opt3.append(crop)
                elif len(opt) == 4:
                    crop = np.zeros(rec4)
                    high, width, channel = rec4
                    crop = img[opt[0][1]:opt[0][1] + high][:, opt[0][0]:opt[0][0] + width]
                    # cv2.imshow("image", crop)
                    # cv2.waitKey(0)
                    opt4.append(crop)
                elif len(opt) == 7:
                    crop = np.zeros(rec7)
                    high, width, channel = rec7
                    crop = img[opt[0][1]:opt[0][1] + high][:, opt[0][0]:opt[0][0] + width]
                    # cv2.imshow("image", crop)
                    # cv2.waitKey(0)
                    opt7.append(crop)
            n += 1
            progress.show_progress(n)
        progress.end()
    return np.array(opt3), np.array(opt4), np.array(opt7)


def save_options(opt3:np.array, opt4: np.array, opt7: np.array):
    with open("options_three.pkl", "wb") as f:
        pickle.dump(opt3, f)
    with open("options_four.pkl", "wb") as f:
        pickle.dump(opt4, f)
    with open("options_seven.pkl", "wb") as f:
        pickle.dump(opt7, f)


# 获取每一题的所有选项位置
option_list = load_data(template_json_dir)

# 计算x选题的平均标准画布
rec3, rec4, rec7 = get_template_rec(option_list)

# 得到options的list
opt3, opt4, opt7 = get_options(option_list, rec3, rec4, rec7, userimg_png_dir)

# 存储
save_options(opt3, opt4, opt7)



