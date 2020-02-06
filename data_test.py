import cv2
import numpy as np
import json

template_json_dir = "data/template.json"
template_png_dir = "data/template-00010001.png"


def load_data(json_dir: str, png_dir: str):
    with open(json_dir, 'r') as load_f:
        load_template = json.load(load_f)
        print(type(load_template))
        print(type(load_template[0]["choice"]))
        # print(load_template[0]["choice"])

        sub_list = load_template[0]["choice"]
        template_img = cv2.imread(png_dir)
        for i in sub_list:
            print("题号：%s   值：%s" % (sub_list.index(i) + 1, i))

            # 把每一题的题号框出来
            rec1 = i
            pt0 = rec1["number"][0]
            pt1 = rec1["number"][1]
            pt2 = rec1["number"][2]
            pt3 = rec1["number"][3]
            cv2.rectangle(template_img, pt1=(pt0, pt1), pt2=(pt2, pt3), color=(255, 0, 0), thickness=1)
        cv2.imshow('image', template_img)
        cv2.waitKey(0)
    return sub_list


# list = load_data(template_json_dir, template_png_dir)

# a = np.zeros((2, 3))
# print(a)
# a[0][1] = 1
# print(a)






