# 이미지를 각각의 patch로 쪼갠 후 분류모델로 어떤 texture인지 구분해 보는 것이다.
# 결과가 좋지는 않았다. 아무래도 학습한 이미지와 테스트하는 이미지 사이에 확대 축소된 비율차이가 커서 그런 것 같기도 하다.
# 학습한 이미지는 비교적 가까운 거리에서 texture를 찍은 것이어서 해상도가 높은 반면
# 테스트에 사용한 이미지는 texture들이 멀리 떨어져 있어서 해상도가 낮아질 수 밖에 없다.
# 그래서 학습을 할 때, 실제 테스트할 이미지와 비슷한 이미지를 patch로 쪼개서 patch들을 학습을 시켜봐야될 것 같다.

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import time
import pickle

def image_read(folder, name, fx=0.25, fy=0.25, gray=False):
    image = cv2.imread(folder + name)
    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def extract_glcm_feature(image):
    glcm = greycomatrix(image, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    list_features = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feature = [greycoprops(glcm, f)[0, 0] for f in list_features]
    return feature

def extract_glcm_feature_RGB(image):
    feature = []
    for i in range(3):  # BGR순으로 feature를 뽑는다.
        feature += extract_glcm_feature(image[:, :, i])
    return feature

# 이미지를 patch로 나누기
# https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array

image = image_read("./images/temp/", "road_scene3.jpg")
cv2.imshow("temp", image)

PATCH_SIZE = 10 #이미지를 블록으로 나눔
height, width, channel = image.shape
# patches = image.reshape(image.shape[0]//PATCH_SIZE, PATCH_SIZE, image.shape[1]//PATCH_SIZE, PATCH_SIZE).swapaxes(1, 2).reshape(-1, PATCH_SIZE, PATCH_SIZE)
patches = image.reshape(height//PATCH_SIZE, PATCH_SIZE, width//PATCH_SIZE, PATCH_SIZE, channel).swapaxes(1, 2).reshape(-1, PATCH_SIZE, PATCH_SIZE, channel)

with open('linearSVC.pickle', 'rb') as f:
    model = pickle.load(f)

mask = np.zeros((height//PATCH_SIZE, width//PATCH_SIZE, channel), np.uint8)
label_color = {
    "road" : np.array([0,0,255]),
    "grass" : np.array([0,255,0]),
    "rocks" : np.array([195,195,195])
}
h = height//PATCH_SIZE
w = width//PATCH_SIZE

print(patches.shape)
print(h, w)

# for i in range(patches.shape[0]):
#     print(i,i%h,i//h)

start = time.time()
for i in range(patches.shape[0]):
    feature = extract_glcm_feature_RGB(patches[i])
    prediction = model.predict(np.array(feature).reshape([1, -1]))  # 분류하기
    print(prediction)
    mask[i%h][i//h] = label_color[prediction[0]]
    cv2.imshow("patch",patches[i])
    cv2.waitKey(2)

end = time.time()
print(end - start)
mask = cv2.resize(mask, None, fx=PATCH_SIZE, fy=PATCH_SIZE, interpolation=cv2.INTER_AREA)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()