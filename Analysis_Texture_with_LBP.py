# Local Binary Pattern을 사용해 Texture를 분석하는 코드이다.

# https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import matplotlib.pyplot as plt


training_images_path = "./images/training/"
testing_images_path = "./images/testing/"
temp_path = "./images/temp/"

# # initialize the local binary patterns descriptor along with
# # the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []


kind_of_labels = os.listdir(training_images_path)
imagePaths = list(map(lambda x:training_images_path + x + "/", kind_of_labels))
print(imagePaths)

for imagePath in os.listdir(temp_path):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(temp_path + imagePath)
    # image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # prediction = model.predict(hist.reshape(1, -1))
    plt.plot(hist)
    plt.show()

    # display the image and the prediction
    # cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #             1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# loop over the training images
# for imagePath in imagePaths:
#     print(imagePath)
#     label = imagePath.split('/')[-2]
#     print(label)
#     for imageName in os.listdir(imagePath):
#         # load the image, convert it to grayscale, and describe it
#         image = cv2.imread(imagePath + imageName)
#         print(imagePath + imageName)
#
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         hist = desc.describe(gray)
#         # extract the label from the image path, then update the
#         # label and data lists
#         labels.append(label)
#
#         data.append(hist)
#
#         plt.plot(hist)
#         plt.show()
#
# # # # train a Linear SVM on the data
# # # model = LinearSVC(C=100.0, random_state=42)
# model = LinearSVC(C=1, random_state=42, max_iter = 10000000)
# model.fit(data, labels)
# #
# # # loop over the testing images
# for imagePath in os.listdir(testing_images_path):
#     # load the image, convert it to grayscale, describe it,
#     # and classify it
#     image = cv2.imread(testing_images_path + imagePath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     hist = desc.describe(gray)
#     prediction = model.predict(hist.reshape(1, -1))
#
#     # display the image and the prediction
#     cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.0, (0, 0, 255), 3)
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)

#------------------------------------------------------------------------
# # import the necessary packages
# from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
# from sklearn.svm import LinearSVC
# from imutils import paths
# import argparse
# import cv2
# import os
#
# # construct the argument parse and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-t", "--training", required=True,
# # 	help="path to the training images")
# # ap.add_argument("-e", "--testing", required=True,
# # 	help="path to the tesitng images")
# # args = vars(ap.parse_args())
#
# training_images_path = "./images/training/"
# testing_images_path = "./images/testing/"
# # initialize the local binary patterns descriptor along with
# # the data and label lists
# desc = LocalBinaryPatterns(24, 8)
# data = []
# labels = []
#
#
# labels = os.listdir(training_images_path)
# imagePaths = list(map(lambda x:training_images_path + x + "/", labels))
# print(imagePaths)
#
# # loop over the training images
# for imagePath in paths.list_images(args["training"]):
# 	# load the image, convert it to grayscale, and describe it
# 	image = cv2.imread(imagePath)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	hist = desc.describe(gray)
# 	# extract the label from the image path, then update the
# 	# label and data lists
# 	labels.append(imagePath.split(os.path.sep)[-2])
# 	data.append(hist)
# # # train a Linear SVM on the data
# # model = LinearSVC(C=100.0, random_state=42)
# # model.fit(data, labels)
# #
# # # loop over the testing images
# # for imagePath in paths.list_images(args["testing"]):
# #     # load the image, convert it to grayscale, describe it,
# #     # and classify it
# #     image = cv2.imread(imagePath)
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     hist = desc.describe(gray)
# #     prediction = model.predict(hist.reshape(1, -1))
# #
# #     # display the image and the prediction
# #     cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
# #                 1.0, (0, 0, 255), 3)
# #     cv2.imshow("Image", image)
# #     cv2.waitKey(0)