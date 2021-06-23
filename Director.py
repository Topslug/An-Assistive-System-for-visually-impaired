import cv2
import numpy as np
from skimage.filters import sobel_h, sobel_v
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

class Director:
    def __init__(self):
        return

    def set_image_size(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        return

    def get_ROI(self, image):#image, 2차원 행렬
        #image를 ROI로 나눈다.
        # 왼쪽/오른쪽 반으로 나누기
        h,w = image.shape

        ROI_left = image[:,:w//2]
        ROI_right = image[:,w//2:]

        return ROI_left, ROI_right

    def get_gradient(self, image):
        dy = sobel_v(image)  # dy 성분, 가로방향 edge, width방향
        dx = sobel_h(image)  # dx 성분, 세로방향 edge, height방향
        return dx, dy

    def get_magnitude(self, dx, dy, is_int=False):
        # Gradient의 크기를 계산하는 함수
        mag = np.sqrt(dx ** 2 + dy ** 2)
        mag = mag/mag.max()# 정규화, 0 ~ 1사이의 값을 가지는 실수값
        if is_int:
            mag = np.uint8(mag*255)
        return mag

    # 두 개 분산의 차이, magnitude 차이가 더 벌어질 것으로 예상했지만 오히려 분산이 줄었다.
    # 분산이 줄었다는 것은 편차가 더 줄었다는 것이므로 좋은 함수가 아니다.
    # 786.2542431327159
    # 775.3633456117947
    def get_magnitude_half(self, dx, dy):
        dx2 = (dx/dx.max())*1.5
        dy2 = (dy/dy.max())*1.5

        mag = np.sqrt(dx2**2 + dy2**2)
        mag = mag/mag.max()
        mag = np.uint8(mag*255)
        return mag

    def get_angle(self, dx, dy, is_int=False):
        # 각도를 계산하는 함수

        # np.arctan가 값이 어떻게 반환되는지 확인해보았다.
        # x = np.array([-1, +1, +1, -1])
        # y = np.array([-1, -1, +1, +1])
        # print(np.arctan2(y, x) * 180 / np.pi)
        # #[-135.  -45.   45.  135.]
        # print(np.mod(np.arctan2(y, x), np.pi)*180/np.pi)
        # #[ 45. 135.  45. 135.]

        """Calculate the angles between horizontal and vertical operators."""
        # np.arctan2(y_coordinate, x_coordinate)
        angle = np.mod(np.arctan2(dy, dx), np.pi)
        angle = np.degrees(angle)

        if is_int:
            angle = angle.astype(np.uint8)

        return angle

    #https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    def non_max_suppression(self, img, angle):# img : image, angle : degree(radian 아님)
        Height, Width = img.shape
        nms_img = np.zeros((Height, Width), np.int32)

        for i in range(1, Height - 1):
            for j in range(1, Width - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i + 1, j + 1]
                        r = img[i - 1, j - 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i - 1, j + 1]
                        r = img[i + 1, j - 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        nms_img[i, j] = img[i, j]
                    else:
                        nms_img[i, j] = 0

                except IndexError as e:
                    pass

        nms_img = nms_img.astype(np.uint8)
        return nms_img

    def get_erode_img(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        erosion = cv2.erode(image, kernel)
        return erosion

    def get_dilate_img(self, image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilation = cv2.dilate(image, kernel)
        return dilation

    def labeling(self, image):
        Height, Width = image.shape

        labeled = np.zeros(image.shape)
        visited = np.zeros(image.shape)

        start_x = Height - 1
        start_y = Width // 2

        dr = [[0,1],[0,-1],[1,0],[-1,0]]

        stack = []
        stack.append((start_x, start_y))

        while len(stack):
            cur_x, cur_y = stack.pop()
            for dx, dy in dr:
                # 경계 확인
                if (cur_x + dx < 0) or (cur_x + dx >= Height) or (cur_y + dy < 0) or (cur_y + dy >= Width):
                    continue
                # 방문 확인
                if labeled[cur_x + dx, cur_y + dy] == 1 or visited[cur_x+dx, cur_y+dy] == 1:
                    continue
                # 값 확인
                if (image[cur_x + dx, cur_y + dy] == 0):
                    labeled[cur_x + dx, cur_y + dy] = 1
                    visited[cur_x + dx, cur_y + dy] = 1
                    stack.append((cur_x + dx, cur_y + dy))
        return labeled

    def wide_labeling(self, image, size=11):
        padding = size//2
        limit = padding + 1

        Height, Width = image.shape

        labeled = np.zeros(image.shape, np.uint8)
        visited = np.zeros(image.shape, np.uint8)

        start_x = Height - limit
        start_y = Width // 2

        dr = [[0, size], [0, -size], [size, 0], [-size, 0]]

        stack = []
        stack.append((start_x, start_y))

        while len(stack):
            cur_x, cur_y = stack.pop()
            for dx, dy in dr:
                # 경계 확인
                if (cur_x + dx - padding < 0) or (cur_x + dx + padding >= Height) or (cur_y + dy - padding < 0) or (cur_y + dy + padding >= Width):
                    continue
                # 방문 확인
                if labeled[cur_x + dx, cur_y + dy] == 1 or visited[cur_x + dx, cur_y + dy] == 1:
                    continue
                patch = image[cur_x-padding:cur_x+limit,cur_y-padding:cur_y+limit]
                # 값 확인
                if (np.sum(patch) <= 30):
                    labeled[cur_x-padding:cur_x+limit,cur_y-padding:cur_y+limit] = 1
                    visited[cur_x-padding:cur_x+limit,cur_y-padding:cur_y+limit] = 1
                    stack.append((cur_x + dx, cur_y + dy))
        return labeled

    def get_road_region_and_save(self, image, image_name):
        print(image_name)
        dx, dy = self.get_gradient(image) # gradient를 구한다음
        magnitude = self.get_magnitude(dx, dy, is_int=True)# 크기를 구한다.
        cv2.imwrite("./images/temp/test_result/"+image_name[:-4]+"_1magnitude.jpg", magnitude)
        #오츠 방법으로 이진화를 한 다음
        ret, thr = cv2.threshold(magnitude, magnitude.min(), magnitude.max(), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite("./images/temp/test_result/" + image_name[:-4] + "_2otsu.jpg", thr)
        median = cv2.medianBlur(thr, 5)# 미디안 필터로 salt & pepper noise를 없앤다.
        cv2.imwrite("./images/temp/test_result/" + image_name[:-4] + "_3median.jpg", median)
        median_dilation = self.get_dilate_img(median)# 그 다음 max filter(팽창연산)으로 영역을 확장 시킨다.마스크 크기 (7,7)
        cv2.imwrite("./images/temp/test_result/" + image_name[:-4] + "_4median_dilation.jpg", median_dilation)
        labeled = self.wide_labeling(median_dilation)#다음 넓은 pixel의 마스크를 이용해 이미지 하단 중간 점 부터 labeling을 수행한다.
        cv2.imwrite("./images/temp/test_result/" + image_name[:-4] + "_5labeled.jpg", labeled*255)
        return labeled

    def get_road_region(self, image):
        dx, dy = self.get_gradient(image) # gradient를 구한다음
        magnitude = self.get_magnitude(dx, dy, is_int=True)# 크기를 구한다.
        #오츠 방법으로 이진화를 한 다음
        ret, thr = cv2.threshold(magnitude, magnitude.min(), magnitude.max(), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        median = cv2.medianBlur(thr, 5)# 미디안 필터로 salt & pepper noise를 없앤다.
        median_dilation = self.get_dilate_img(median)# 그 다음 max filter(팽창연산)으로 영역을 확장 시킨다.마스크 크기 (7,7)
        labeled = self.wide_labeling(median_dilation)#다음 넓은 pixel의 마스크를 이용해 이미지 하단 중간 점 부터 labeling을 수행한다.
        return labeled

    # def get_EDF(self, magnitude, angle):
    def get_EDF(self, image, relative=True):
        # magnitude : gradient magnitude image, Edge Distribution Function
        # angle : gradient의 방향
        # EDF를 구하는 함수 이다.

        dx, dy = self.get_gradient(image)
        angle = self.get_angle(dx, dy, is_int=True)# 0~180 사이의 정수값
        magnitude = self.get_magnitude(dx, dy)

        threshold = np.quantile(magnitude, 0.97)#97%에 해당하는 magnitude를 임계값으로 잡는다.
        magnitude[magnitude < threshold] = 0

        edf = np.zeros(180)
        for a in range(180):
            edf[a] = np.sum(magnitude[angle == a])
        edf[0] += np.sum(magnitude[angle == 180])

        if relative: # relative value 획득
            edf = edf/np.sum(edf)

        return magnitude, angle, edf#magnitude는 실수값, angle은 정수값, edf는 실수값

    def analysis_EDF(self, edf):  # 가장 세기가 센 p와 theta값 획득
        # 먼저 가우시안 필터로 부드럽게 만든다.
        kernel = cv2.getGaussianKernel(13, 2)
        kernel = kernel.reshape(1, -1)[0]
        smooth = ndi.filters.convolve1d(edf, kernel)

        p = np.max(smooth)  # 최대값 선택
        theta = np.argmax(smooth)  # 최대값이 되는 밝기의 각도 선택

        if p == 0:
            theta = 180

        return p, theta, smooth

    def decision_making(self, theta_left, theta_right):

        slope_left = np.tan(np.radians(theta_left))
        slope_right = np.tan(np.radians(theta_right))

        c = slope_left * slope_right
        if c < 0:
            return "Going well!"
        else:# c가 양수이거나 0인 경우
            if slope_left > 0 or slope_right > 0:
                return "Turn right!"
            else:
                return "Turn left!"

    def estimate_line(self, image, angle, EDF):

        Height = image.shape[0]
        Width = image.shape[1]

        line_img = np.zeros((Height, Width), np.uint8)

        try:
            theta = EDF.argmax()
            idx = np.where(angle == theta)

            X = idx[0]#세로축 값
            Y = idx[1]#가로축 값

            line_img[X,Y] = 255

            X_mean = X.mean()
            Y_mean = Y.mean()

            slope = np.tan(np.radians(theta - 90))

            b = Y_mean - slope * X_mean

            p1 = (int(b), 0)
            p2 = (int(slope*Height + b), Height)

            cv2.line(line_img, p1, p2, (255, 255, 255), 3)
        except:
            return line_img

        return line_img

    def estimate_line_with_random_point(self, image, magnitude, angle, EDF, n_points = 100): # magnitude는 실수값, angle은 정수값, EDF는 실수값

        Height = image.shape[0]
        Width = image.shape[1]

        theta = EDF.argmax()
        idx = np.where(angle == theta)
        selected = np.random.choice(len(idx[0]), n_points, replace=False)

        X = idx[0][selected]# numpy 객체
        Y = idx[1][selected]

        X_mean = X.mean()
        Y_mean = Y.mean()

        A = np.vstack([X - X_mean, Y - Y_mean]).T

        # https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=220885732785
        U, sigma, VT = np.linalg.svd(A)#sigma 배열

        a,b = VT[-1]
        d = a*X_mean + b*Y_mean

        x_data = np.linspace(0, Height, Height)
        y_data = -(a/b) * x_data + d

        print(a, b, a ** 2 + b ** 2)

        line_img = np.zeros((Height, Width))
        # ((d - a*Height)/b, Height)
        # (d/b, 0)
        cv2.line(line_img,(int(d/b), 0),(int((d - a*Height)/b), Height),(255,255,255),3)

        cv2.imshow("line image", line_img)

        plt.plot(x_data, y_data)
        plt.show()

        return


if __name__ == "__main__":

    kind_of_test = "image"
    # kind_of_test = "video"

    if kind_of_test == "video":
        import time
        cap = cv2.VideoCapture("./images/walking video5.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_H, out_W = 480, 1300
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (out_W, out_H))

        director = Director()

        while True:

            start = time.time()
            success, image = cap.read()
            if not success:
                print("end")
                break

            original = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape
            image = image[h//3:,:]
            image = cv2.resize(image, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_AREA)
            image = cv2.GaussianBlur(image, (19, 19), 3)

            # ROI 분리
            image_left, image_right = director.get_ROI(image)

            # 각각 magnitude와 angle, EDF를 구한다.
            magnitude_left, angle_left, EDF_left = director.get_EDF(image_left, relative=True)
            p_left, theta_left, smooth_left = director.analysis_EDF(EDF_left)

            magnitude_right, angle_right, EDF_right = director.get_EDF(image_right, relative=True)
            p_right, theta_right, smooth_right = director.analysis_EDF(EDF_right)

            # 왼쪽으로 돌지, 오른쪽으로 돌지, 똑바로 갈지 결정한다.
            decision = director.decision_making(theta_left, theta_right)

            # Road 영역 분리, Road Segmentation
            road_region = director.get_road_region(image)

            #line 추정
            right_line = director.estimate_line(image_right, angle_right, EDF_right)
            left_line = director.estimate_line(image_left, angle_left, EDF_left)

            #출력
            cv2.putText(original, decision, (0, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 125), 2)
            cv2.putText(original, "FPS : {:.2f}".format(1 / (time.time() - start)),
                        (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 125, 125), 2)
            # cv2.imshow("original", original)
            # cv2.imshow("left & right ROI magnitude", np.hstack([magnitude_left, magnitude_right]))
            # cv2.imshow("left & right line", np.hstack([left_line, right_line]))
            # cv2.imshow("road region", road_region * 255)

            merged_frame = np.vstack(
                [np.hstack([magnitude_left, magnitude_right])*255, np.hstack([left_line, right_line]), road_region * 255])
            merged_frame = merged_frame.astype(np.uint8)
            merged_frame = np.concatenate([original, cv2.cvtColor(merged_frame, cv2.COLOR_GRAY2BGR)], axis=1)
            padding = np.zeros((out_H, out_W - merged_frame.shape[1], 3), np.uint8)
            out_frame = np.concatenate([padding, merged_frame], axis=1)
            cv2.imshow("result", out_frame)

            out.write(out_frame)
            # cv2.imshow("result", merged_frame)
            cv2.waitKey(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


    elif kind_of_test == "image":
        import os
        import time
        image_list = os.listdir("./images/temp/director/")[:-1]
        print(image_list)

        WIDTH = 720  #0.5
        HEIGHT = 540 #0.5

        director = Director()

        for image_name in image_list:
            image = cv2.imread("./images/temp/director/" + image_name)

            print(image_name)
            # preprocessing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imshow("original", image)
            image = cv2.GaussianBlur(image, (19, 19), 3)

            #ROI 분리
            image_left, image_right = director.get_ROI(image)
            cv2.imshow("right", image[:,WIDTH//2:])
            #magnitude, angle, EDF = director.get_EDF(image, relative=True)

            #각각 magnitude와 angle, EDF를 구한다.
            magnitude_left, angle_left, EDF_left = director.get_EDF(image_left, relative=True)
            p_left, theta_left, smooth_left = director.analysis_EDF(EDF_left)

            magnitude_right, angle_right, EDF_right = director.get_EDF(image_right, relative=True)
            cv2.imshow("right image", image_right)
            p_right, theta_right, smooth_right = director.analysis_EDF(EDF_right)

            # 왼쪽으로 돌지, 오른쪽으로 돌지, 똑바로 갈지 결정한다.
            decision = director.decision_making(theta_left, theta_right)
            print(decision)

            # Road 영역 분리, Road Segmentation
            road_region = director.get_road_region(image)
            # road_region = director.get_road_region_and_save(image, image_name)

            right_line = director.estimate_line(image_right, angle_right, EDF_right)
            left_line = director.estimate_line(image_left, angle_left, EDF_left)

            cv2.imshow("magnitude merged ", np.hstack([magnitude_left, magnitude_right, ]))
            # cv2.imshow("magnitude ", magnitude)
            cv2.imshow("road region", road_region*255)
            cv2.imshow("line image", np.hstack([left_line,right_line]))

            fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
            ax[0, 0].plot(EDF_left)
            ax[1, 0].plot(smooth_left)
            ax[0, 1].plot(EDF_right)
            ax[1, 1].plot(smooth_right)

            plt.tight_layout()
            plt.show()
            # plt.savefig("./images/temp/test_result/"+image_name[:-4]+"_EDF_left_right.jpg")
            # plt.cla()
            cv2.waitKey(1)
        cv2.destroyAllWindows()
