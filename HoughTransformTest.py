# https://stackoverflow.com/questions/57535865/extract-vanishing-point-from-lines-with-open-cv
# hough line을 추정한 후에 vanishing point를 찾는 방법이다.
# 위 사이트에서 답변해준 사람은 추출된 선을 억제하고 팽창시키면 소실점이 나온다고 설명해 주었다.
# 하지만 hough line이 잘 검출 되었을 때 그렇게 할 수 있는 것 같다.
# blur를 많이 줌으로써 노이즈를 많이 제거했는데도 엉뚱한 hough line이 많이 발생하여 사용하기 적합하지 않았다.

import cv2
import numpy as np

img = cv2.imread("./images/temp/road_scene2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

## ROI 선택
h, w = gray.shape
# cv2.imshow("roi",gray[int(h/3):,:])
##

blurred = cv2.GaussianBlur(gray, (35,35), 15)
cv2.imshow("blurred", blurred)

edges = np.zeros((h, w), np.uint8)
edges[int(h/3):,:] = cv2.Canny(gray[int(h/3):,:],50,250,apertureSize = 3)

# fine tune parameters
lines = cv2.HoughLines(edges, 0.7, np.pi/120, 120, min_theta=np.pi/36, max_theta=np.pi-np.pi/36)
line_image = np.zeros(gray.shape)
for line in lines:
    rho,theta = line[0]
    # skip near-vertical lines
    if abs(theta-np.pi/90) < np.pi/9:
        continue
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),1)

# lines = cv2.HoughLines(edges,1,np.pi/180,250)
# line_image = np.zeros(gray.shape)
#
# for line in lines:
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),1)

# erode : 침식 선폭을 줄인다.
# dilate : 팽창 선폭을 늘린다.
# delete lines
kernel = np.ones((3,3),np.uint8)
img2 = cv2.erode(line_image,kernel,iterations = 1)
# strengthen intersections
kernel = np.ones((9,9),np.uint8)
img2 = cv2.dilate(img2,kernel,iterations = 1)
# close remaining blobs
kernel = np.ones((11,11),np.uint8)
img2 = cv2.erode(img2,kernel,iterations = 1)
img2 = cv2.dilate(img2,kernel,iterations = 1)
# cv2.imwrite('points.jpg', img2)

cv2.imshow('edges', edges)
cv2.imshow('result', img)
cv2.imshow('line image', line_image)
cv2.imshow('vanishing point', img2)
cv2.waitKey()
cv2.destroyAllWindows()