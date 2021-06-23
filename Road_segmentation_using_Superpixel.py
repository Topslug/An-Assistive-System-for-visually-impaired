# Camera-based On-Road Detections for the Visually Impaired, Judith Jakob1,2, József Tick3
# 논문을 구현한 결과이다.
# 논문에는 횡단보도를 검출하는 부분이 있긴 했지만, 이해가 잘 안되기도 하고,
# 내가 테스트할 환경에는 횡단보도가 없기 때문에 구현해보지는 않았다.
# 대신 도로를 분할하는 과정과 차선을 검출하는 부분은 도움이 될 것이라 판단하여 구현해보았다.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.segmentation import watershed
from skimage.filters import sobel, sobel_h, sobel_v
from skimage import segmentation, color
from skimage.future import graph
from scipy import ndimage as ndi


def get_mosaic(image, segments):
    # segment로 나뉘어진 image를 segment별로 평균을 내는 함수이다.
    # segments에는 label로 영역이 분할되어 있다.
    mosaic = np.zeros(image.shape)
    for i in range(len(np.unique(segments))):
        mosaic[segments == i] = np.mean(image[segments == i], axis=0)
    mosaic = mosaic.astype(np.uint8)
    return mosaic

def get_superpixel_slic_opencv(image, region_size=12, iteration=10):
    # opencv에 있는 slic알고리즘을 이용하여 superpixel을 구하는 함수이다.
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0)
    slic.iterate(iteration)
    segments = slic.getLabels()

    output = get_mosaic(image, segments)
    return output

def get_superpixel_slic_skimage(image, compactness=30, n_segments=200):
    # skimage에 있는 slic알고리즘으로 superpixel을 구한다.
    segments = segmentation.slic(image, compactness=compactness, n_segments=n_segments, start_label=1)
    output = color.label2rgb(segments, image, kind='avg', bg_label=0)
    output = output.astype(np.uint8)
    return output

def get_superpixel_watershed_skimage(image, gray=False, markers=250, compactness=0.001):
    #skimage에 있는 watershed함수로 superpixel을 구한다.
    if not gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient = sobel(image)
    segments = watershed(gradient, markers=markers, compactness=compactness)
    output = color.label2rgb(segments, image, kind='avg', bg_label=0)
    output = output.astype(np.uint8)
    return output

def merge_superpixel_slic_opencv(image, region_size=15, iteration=10, threshold=20):
    #superpixel구하기
    slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0)
    slic.iterate(iteration)
    segments = slic.getLabels()

    # superpixel 합치기
    g = graph.rag_mean_color(image, segments)
    merged_segments = graph.cut_threshold(segments, g, threshold)
    output = get_mosaic(image, merged_segments)
    return output

def merge_superpixel_slic_skimage(image, compactness=30, n_segments=200, threshold=15):
    # superpixel로 분할 된 이미지를 threshold를 이용하여 일정 차이보다 작으면 병합하는 함수 있다.
    segments = segmentation.slic(image, compactness=compactness, n_segments=n_segments, start_label=1)

    g = graph.rag_mean_color(image, segments)
    merged_segments = graph.cut_threshold(segments, g, threshold)
    output = color.label2rgb(merged_segments, image, kind='avg', bg_label=0)
    output = output.astype(np.uint8)

    return output

def merge_superpixel_watershed_skimage(image, gray=False, markers=250, compactness=0.001, threshold=15):
    if not gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient = sobel(image)
    segments = watershed(gradient, markers=markers, compactness=compactness)

    g = graph.rag_mean_color(image, segments)
    merged_segments = graph.cut_threshold(segments, g, threshold)
    output = get_mosaic(image, merged_segments)
    return output

def show_animation_segments(image, algorithm="slic_opencv", wait=1, region_size=15, compactness=30, n_segments=200):
    if algorithm == "slic_opencv":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0)
        slic.iterate(iteration)
        segments = slic.getLabels()
    if algorithm == "slic_skimage":
        segments = segmentation.slic(image, compactness=compactness, n_segments=n_segments, start_label=1)
    elif algorithm == "watershed":#color image라고 가정
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient = sobel(image)
        segments = watershed(gradient, markers=250, compactness=0.001)

    num_segments = len(np.unique(segments))
    for i in range(num_segments):
        mask = (segments == i).astype(np.uint8) * 255
        patch = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("patch", patch)
        cv2.waitKey(wait)

# Road scene
# #slic opencv
# region_size_opencv = 15
# iteration_opencv = 10
# threshold_slic_opencv = 20
#
# #slic skimage
# n_segments_skimage = 200
# compactness_skimage = 30
# threshold_skimage = 15
#
# #watershed skimage
# markers = 250
# compactness_watershed = 0.001
# threshold_watershed = 15


# Road인 경우 #
#slic opencv
region_size_opencv = 15
iteration_opencv = 10
threshold_slic_opencv = 20

#slic skimage
n_segments_skimage = 500
compactness_skimage = 30
threshold_skimage = 13

#watershed skimage
markers = 500
compactness_watershed = 0.001
threshold_watershed = 13

fx = 1
fy = 1

select_algorithm = "slic_skimage"
# select_algorithm = "slic_opencv"
# select_algorithm = "watershed_skimage"
###

# image = cv2.imread("./images/temp/road_scene3.jpg")
# fx = 0.25
# fy = 0.25
image = cv2.imread("./images/temp/road.jpg")



image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
image = cv2.GaussianBlur(image, (3,3) ,1)
# cv2.imshow("image", image)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_for_opencv = image_gray

start = time.time()
superpixel_slic_opencv = get_superpixel_slic_opencv(image_gray, region_size=region_size_opencv, iteration=iteration_opencv)
end = time.time()
print("opencv slic time : ", end - start)
print("FPS : ", 1/(end - start))

start = time.time()
merged_superpixel_slic_opencv = merge_superpixel_slic_opencv(image_gray, region_size=region_size_opencv, iteration=iteration_opencv, threshold=threshold_slic_opencv)
end = time.time()
print("opencv merged superpixel time : ", end - start)
print("FPS : ", 1/(end - start))

start = time.time()
superpixel_slic_skimage = get_superpixel_slic_skimage(image, compactness=compactness_skimage, n_segments=n_segments_skimage)
end = time.time()
print("skimage slic time : ",end - start)
print("FPS : ", 1/(end - start))


start = time.time()
merged_superpixel_slic_skimage = merge_superpixel_slic_skimage(image, compactness=compactness_skimage, n_segments=n_segments_skimage, threshold=threshold_skimage)
end = time.time()
print("skimage slic merged superpixel time : ",end - start)
print("FPS : ", 1/(end - start))

start = time.time()
superpixel_watershed_skimage = get_superpixel_watershed_skimage(image, gray=False, markers=markers, compactness=compactness_watershed)
end = time.time()
print("skimage watershed time : ", end - start)
print("FPS : ", 1/(end - start))

start = time.time()
merged_superpixel_watershed_skimage = merge_superpixel_watershed_skimage(image, gray=False, markers=markers, compactness=compactness_watershed, threshold=threshold_watershed)
end = time.time()
print("skimage watershed merged superpixel time : ", end - start)
print("FPS : ", 1/(end - start))

fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

ax[0, 0].imshow(superpixel_slic_opencv)#make_boundaries
ax[0, 0].set_title("opencv slic")
ax[1, 0].imshow(merged_superpixel_slic_opencv)
ax[1, 0].set_title('opencv slic merged')

ax[0, 1].imshow(superpixel_slic_skimage)
ax[0, 1].set_title('skimage slic')
ax[1, 1].imshow(merged_superpixel_slic_skimage)
ax[1, 1].set_title('skimage slic merged')

ax[0, 2].imshow(superpixel_watershed_skimage)
ax[0, 2].set_title('skimage watershed')
ax[1, 2].imshow(merged_superpixel_watershed_skimage)
ax[1, 2].set_title('skimage watershed merged')

# cv2.imshow("mosaic_segments_slic_opencv",mosaic_segments_slic_opencv)
# cv2.imshow("mosaic_merged_segments_slic_opencv",mosaic_merged_segments_slic_opencv)
# cv2.imshow("mosaic_segments_slic_skimage",mosaic_segments_slic_skimage)
# cv2.imshow("mosaic_merged_segments_slic_skimage",mosaic_merged_segments_slic_skimage)
# cv2.imshow("mosaic_segments_watershed_skimage",mosaic_segments_watershed_skimage)
# cv2.imshow("mosaic_merged_segments_watershed_skimage",mosaic_merged_segments_watershed_skimage)

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
# plt.savefig("./superpixel_test.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
#---------------------------------------
# Lane Detection
# https://opencv-python.readthedocs.io/en/latest/doc/19.imageHistograms/imageHistograms.html

def get_road_region(input):
    hist = cv2.calcHist([input], [0], None, [256], [0, 256])
    max_value = np.argmax(hist)
    road_region = (input == max_value)
    return road_region

def make_road_region_mask(road_region):
    mask = road_region.astype(np.uint8)
    mask *= 255
    return mask

def get_ROI(image, road_region):
    temp = np.sum(road_region, axis=1)
    min_y = np.sum(temp == 0)
    return min_y, image[min_y:,:].copy()

# https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html
# https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html

# np.arctan가 값이 어떻게 반환되는지 확인해보았다.
# x = np.array([-1, +1, +1, -1])
# y = np.array([-1, -1, +1, +1])
# print(np.arctan2(y, x) * 180 / np.pi)
# #[-135.  -45.   45.  135.]
# print(np.mod(np.arctan2(y, x), np.pi)*180/np.pi)
# #[ 45. 135.  45. 135.]
def get_angle(dx, dy):
    # 각도를 계산하는 함수
    """Calculate the angles between horizontal and vertical operators."""
    #np.arctan2(y_coordinate, x_coordinate)
    output = np.mod(np.arctan2(dy, dx), np.pi)
    output = np.degrees(output)
    return np.round(output)

def get_magnitude(dx, dy):
    # Gradient의 크기를 계산하는 함수
    return np.sqrt(dx**2 + dy**2)

def split_8subimages(input, interval):
    # 주어진 image를 8개의 조각으로 나눈다.
    # 그런데 아래쪽에 있는 이미지를 크게 하고, 위쪽에 있는 이미지는 작게 한다.
    if len(input.shape) == 3:
        h, w, _ = input.shape
    else:
        h, w = input.shape

    images = []
    for i, val in enumerate(interval[:-1]):
        start = h - int(interval[i + 1]*h)# int(interval[i + 1]*h) #h - int(interval[i + 1]*h)
        end = h - int(interval[i]*h) # int(interval[i]*h) #h - int(interval[i]*h)
        images.append(input[start:end, :])
    return images

def get_EDF(input):#input : grayscale image, Edge Distribution Function
    # EDF를 구하는 함수 이다.
    dy = sobel_v(input)  # dy 성분, x방향 edge
    dx = sobel_h(input)  # dx 성분, y방향 edge

    angle = get_angle(dx, dy)
    magnitude = get_magnitude(dx, dy)

    threshold = np.quantile(magnitude, 0.97)
    magnitude[magnitude < threshold] = 0

    edf = np.zeros(180 + 1)
    for a in range(1, 181):
        edf[a] = np.sum(magnitude[angle == a])

    return edf

#1차원 컨볼루션
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.convolve1d.html
def analysis_EDF(edf):# p와 theta값 획득
    kernel = cv2.getGaussianKernel(15, 2.5)
    kernel = kernel.reshape(1, -1)[0]
    smooth = ndi.filters.convolve1d(edf, kernel)

    smooth /= np.sum(smooth)# relative value
    p = np.max(smooth)#최대값 선택
    theta = np.argmax(smooth)#최대값이 되는 밝기의 각도 선택
    print(theta)

    if p == 0:
        theta = 180

    return p, theta

def get_EDF_info(subimages):
    EDF = []
    thetas = []
    p_values = []

    for subimage in subimages:
        edf = get_EDF(subimage)
        EDF.append(edf)

        p, theta = analysis_EDF(edf)
        p_values.append(p)
        thetas.append(theta)
    return EDF, p_values, thetas

def make_y_values(x_values, thetas):
    y_values = [0]
    for i in range(1, 9):  # 0~8중 1~8
        y_value = y_values[i - 1] + (x_values[i] - x_values[i - 1]) * np.tan(np.radians(90 - thetas[i - 1]))
        y_values.append(y_value)
    return y_values

def interpolate_linear_parabolic(x_values, y_values):

    # y = Ap,
    # y = a*(x - 0.5) + b + c*(x-0.5)**2 추정
    # A's row = [(x - 0.5) 1 (x-0.5)**2]
    # p = [a b c], column vector

    #Data Matrix A만들기
    col1 = np.array([(x - 0.5) for x in x_values])  # (x - 0.5)
    col2 = np.ones(9)
    col3 = np.array([(x - 0.5) ** 2 for x in x_values])  # (x-0.5)**2
    col3[:3] = np.zeros(3)
    A = np.vstack([col1, col2, col3]).T

    #Least Squares Method로 파라미터 추정
    a, b, c = np.linalg.lstsq(A, y_values, rcond=None)[0]

    return a, b, c

def draw_EDF(EDF):
    fig, ax = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
    for i, edf in enumerate(EDF):
        ax[i//4, i%4].plot(edf)
        ax[i//4, i%4].set_title("j = {}".format(i+1))

    plt.tight_layout()
    plt.show()
    return

def draw_estimation(parameters, x_values, y_values, num_data=1000,):
    a,b,c = parameters
    plt.scatter(y_values, x_values)
    x = np.linspace(0, 1, num_data)

    half = num_data//2
    y = np.zeros(num_data)
    y[:half] = a * (x[:half] - 0.5) + b
    y[half:] = a * (x[half:] - 0.5) + b + c * (x[half:] - 0.5) ** 2

    plt.plot(y, x)
    plt.show()
    return

def make_decision(p_values, thetas, parameters, thr_p_low=0.014, thr_p_high=0.5 , thr_c_low=0.1,thr_c_mid=2.5, thr_c_high=5, thr_var_low=0.1, thr_var_high=5):
    a,b,c = parameters
    c_abs = np.abs(c)
    mean_p = np.mean(p_values)
    mean_theta = np.mean(thetas)
    var_theta = np.sum((thetas - mean_theta)**2)/7

    print(mean_p)
    if mean_p <= thr_p_low or mean_p > thr_p_high:
        print("No lane detected")
        return

    if c_abs < thr_c_low:
        print("Straight ahead")
        return
    elif thr_c_low < c and c < thr_c_mid:
        if var_theta < thr_var_high:
            print("Straight ahead")
            return
        elif var_theta >= thr_var_high and c > 0:
            print("Left turn")
            return
        else:
            print("Right turn")
            return
    elif thr_c_mid <= c and c < thr_c_high:
        if var_theta < thr_var_low:
            print("Straight ahead")
            return
        elif var_theta >= thr_var_low and c > 0:
            print("Left turn")
            return
        else:
            print("Right turn")
            return
    elif c_abs >= thr_c_high:
        if c > 0:
            print("Left turn")
            return
        else:
            print("Right turn")
            return
    return

if select_algorithm == "slic_opencv":
    input = merged_superpixel_slic_opencv
elif select_algorithm == "slic_skimage":
    input = merged_superpixel_slic_skimage
elif select_algorithm == "watershed_skimage":
    input = merged_superpixel_watershed_skimage

if len(input.shape) == 3:
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

road_region = get_road_region(input)
min_y, ROI = get_ROI(image_gray, road_region)
cv2.imshow("ROI", ROI)
temp = road_region.astype(np.uint8)
temp *= 255
cv2.imshow("road region", temp)

dy = sobel_v(image_gray[min_y:,:]) # dy 성분, x방향 edge
dx = sobel_h(image_gray[min_y:,:]) # dx 성분, y방향 edge
magnitude = get_magnitude(dx, dy)
angle = get_angle(dx, dy)
cv2.imshow("magnitude", magnitude)

x_values = [0, 0.25, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1]
# x_values = [1, 0.9375, 0.875, 0.8125, 0.75, 0.625, 0.5, 0.25, 0]
# x_values = [1 - x for x in x_values]
subimages = split_8subimages(ROI, x_values)

EDF, p_values, thetas = get_EDF_info(subimages)

y_values = make_y_values(x_values, thetas)
parameters = interpolate_linear_parabolic(x_values, y_values)
a,b,c = parameters
print("parameters : ",a,b,c)
draw_EDF(EDF)
draw_estimation(parameters, x_values, y_values)
# make_decision(p_values, thetas, parameters)