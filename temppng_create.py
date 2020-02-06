import cv2
import pickle

img_a = cv2.imread("a.png")
img_b = cv2.imread("b.png")
img_c = cv2.imread("c.png")

# cv2.imshow("image", img_a[1111:1111 + 14][:, 854:854 + 231])
# cv2.waitKey(0)

with open("3-a.pkl", "wb") as f:  # 3选题，A选项
    pickle.dump(img_a[828:828 + 14][:, 388:388 + 110], f)
with open("3-b.pkl", "wb") as f:  # 3选题，B选项
    pickle.dump(img_a[864:864 + 14][:, 388:388 + 110], f)
with open("3-c.pkl", "wb") as f:  # 3选题，C选项
    pickle.dump(img_a[899:899 + 14][:, 388:388 + 110], f)


with open("4-a.pkl", "wb") as f:  # 4选题，A选项
    pickle.dump(img_a[1112:1112 + 14][:, 622:622 + 149], f)
with open("4-b.pkl", "wb") as f:  # 4选题，B选项
    pickle.dump(img_a[1076:1076 + 14][:, 622:622 + 149], f)
with open("4-c.pkl", "wb") as f:  # 4选题，C选项
    pickle.dump(img_a[1184:1184 + 14][:, 389:389 + 149], f)
with open("4-d.pkl", "wb") as f:  # 4选题，D选项
    pickle.dump(img_a[969:969 + 14][:, 621:621 + 149], f)


with open("7-a.pkl", "wb") as f:  # 7选题，A选项
    pickle.dump(img_a[1004:1004 + 14][:, 854:854 + 231], f)
with open("7-b.pkl", "wb") as f:  # 7选题，B选项
    pickle.dump(img_a[968:968 + 14][:, 852:852 + 231], f)
with open("7-c.pkl", "wb") as f:  # 7选题，C选项
    pickle.dump(img_a[1040:1040 + 14][:, 854:854 + 231], f)
with open("7-d.pkl", "wb") as f:  # 7选题，D选项
    pickle.dump(img_c[1111:1111 + 14][:, 854:854 + 231], f)
with open("7-e.pkl", "wb") as f:  # 7选题，E选项
    pickle.dump(img_a[1076:1076 + 14][:, 854:854 + 231], f)
with open("7-f.pkl", "wb") as f:  # 7选题，F选项
    pickle.dump(img_a[1111:1111 + 14][:, 854:854 + 231], f)
with open("7-g.pkl", "wb") as f:  # 7选题，G选项
    pickle.dump(img_b[1111:1111 + 14][:, 854:854 + 231], f)