import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

img_name = "1.jfif"



img = cv2.imread(img_name)
img_length=img.shape[0]
img_width=img.shape[1]
down_sample_factor=1
img = cv2.resize(img,(int(img_width*down_sample_factor),int(img_length*down_sample_factor)))

rows, cols, channel = img.shape

#绕图像的中心旋转
#参数：旋转中心 旋转度数 scale
M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
#参数：原始图像 旋转参数 元素图像宽高
img = cv2.warpAffine(img, M, (cols, rows))


start_time = time.time()
# M = cv2.getRotationMatrix2D((img_width/2,img_length/2), -30, 1.0) #15
# img = cv2.warpAffine(img, M, (img_width, img_length)) #16

    # sift = cv2.xfeatures2d.SIFT_create()#创建sift生成器
    # kp, des = sift.detectAndCompute(img,None)#kp是关键点坐标，des是描述子



detector = cv2.ORB_create()#创建orb生成器
kp = detector.detect(img, None)#kp是关键点坐标
_, des = detector.compute(img, kp)#des是描述子
end_time = time.time()
print("time in ORB generation took %f s"%(end_time-start_time))


print("des shape = ",des.shape)
print("des = ",des)
img_with_kp = cv2.drawKeypoints(img,kp,img,color=(255,0,255)) #画出特征点，并显示为红色圆圈

# cv2.namedWindow("image with keypoint",0);
# cv2.resizeWindow("image with keypoint", img_width, img_length);
cv2.imshow("image with keypoint", img_with_kp) #拼接显示为gray
cv2.waitKey(0)

cv2.destroyAllWindows()