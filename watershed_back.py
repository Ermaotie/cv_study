# coding=utf-8
# 导入相应的python包
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2

# 设置并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# 读取图片
image = cv2.imread(args["image"])
# 进行mean shift滤波
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.namedWindow('Input',cv2.WINDOW_NORMAL)  #创建窗口
cv2.resizeWindow("Input", 1200, 800)  #调整窗口大小
cv2.imshow("Input", image)

# 进行灰度化处理
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
# 进行阈值分割
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.namedWindow('Thresh',cv2.WINDOW_NORMAL)  #创建窗口
cv2.resizeWindow("Thresh", 1200, 800)  #调整窗口大小
cv2.imshow("Thresh", thresh)

# 计算从每个二元像素到最近零像素的精确欧几里得距离，然后在距离图中找到峰值
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

# 利用8连通性对局部峰进行连通分量分析，然后应用分水岭算法
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# 循环显示标签
for label in np.unique(labels):
    # 如果该标签为0，则表示其为背景，直接忽略
    if label == 0:
        continue

    # 为标签区域分配内存并将在mask上绘制结果
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # 在mask上检测轮廓并获得最大的一个轮廓
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # 画一个圈把物体围起来
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 显示最终结果
cv2.namedWindow('Output',cv2.WINDOW_NORMAL)  #创建窗口
cv2.resizeWindow("Output", 1200, 800)  #调整窗口大小
cv2.imshow("Output", image)
cv2.waitKey(0)
