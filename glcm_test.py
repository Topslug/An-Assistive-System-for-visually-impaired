# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.graycoprops
# road_scene 이미지에서 특정 점에서의 glcm을 분석하는 코드이다.

import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
import os


PATCH_SIZE = 21 #이미지를 블록으로 나눔

# open the camera image
#image = data.camera() # sample image를 불러옴

image_folder = "./images/temp/"
image_name = "road_scene3.jpg"
image = cv2.imread(image_folder + image_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("temp", image)

# select some patches from grassy areas of the image 몇개의 patch를 선택
grass_locations = [(540,60), (757,1167), (427,235), (853,1303)]
grass_patches = []
for loc in grass_locations:# grass_locations에 있는 위치에서 patch size만큼 slice를 하여 grass_patches에 저장
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image # sky도 마찬가지로 진행
road_locations = [(827, 241), (905, 725), (685, 523), (557, 787)]
road_patches = []
for loc in road_locations:
    road_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch 각 patch마다 glcm을 계산한다.
xs = []
ys = []
for patch in (grass_patches + road_patches):
    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    # distance : pixel 쌍이 떨어져있는 정도
    # angle : pixel 쌍이 이루는 각도
    # levels : 이미지의 gray level, 양자화를 진행하는 부분
    # symmetric : output을 대칭행렬로 할 것인지 아닌지, True로 하면 계산할 때
    #             pixel밝기 순서를 고려하지 않는다. (i,j) = (j,i)를 똑같이 카운팅
    # normed : 정규화를 할 것인지 말 것인지, 정규화를 하면 output을 전체 pixel개수로 나눈다.
    # return : 4-D ndarray
    # The gray-level co-occurrence histogram.
    # The value P[i,j,d,theta] is the number of times that gray-level j occurs at a distance d and at an angle theta from gray-level i.
    # If normed is False, the output is of type uint32,
    # otherwise it is float64. The dimensions are: levels x levels x number of distances x number of angles.
    # d만큼 떨어져 있고, a의 각도를 이루고 있는 i밝기와 j밝기가 몇 번 등장했는지 4차원 Matrix를 반환

    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    # image와 properties
    # properties는 {‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}중에서 선택
    # 2-dimensional array. results[d, a] is the property ‘prop’ for the d’th distance and the a’th angle.
    # d거리만큼 떨어져 있고, a각도를 이루고 있는 것이 어떤 성질을 가지고 있는지
    # 대비, 차이점, 동질성, 에너지(균일성), 공관계, ASM
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)# 3개 행과 2개의 열이 있는 그리드의 1번째 그림
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in road_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Grass')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Road')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))

for i, patch in enumerate(road_patches):
    ax = fig.add_subplot(3, len(road_patches), len(road_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Road %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()

plt.savefig("./glcm_test_image.jpg")
plt.show()
