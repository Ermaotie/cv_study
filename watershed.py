# coding=utf-8
# 导入相应的python包
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
import imutils
import cv2
import os

# 带有Shift
def watershed_self(input, out=None):
    # 读取图片
    image = cv2.imread(input)
    copyimg = image.copy()
    copyimg = cv2.cvtColor(copyimg,cv2.COLOR_BGR2RGB)
    # 进行mean shift滤波
    shifted = cv2.pyrMeanShiftFiltering(copyimg, 21, 51)

    # 进行灰度化处理
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    # 进行阈值分割
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

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
        cv2.circle(copyimg, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(copyimg, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    src_list = [image, ]
    # 显示最终结果
    fig, ax = plt.subplots(2, 2)
    axes = ax.flatten()
    for axis in axes:
        axis.axis('off')
    axes[0].imshow(image[:, :, ::-1])
    axes[0].set_title("Origin")
    axes[1].imshow(shifted)
    axes[1].set_title("Shifted")
    axes[2].imshow(thresh)
    axes[2].set_title("Thresh")
    axes[3].imshow(copyimg)
    axes[3].set_title("OUT")
    if out is None:
        plt.show()
    else:
        plt.savefig(out)


def main():
    for i, j, k in os.walk(path):
        if not os.path.exists(outpath):
            print("路径不存在，正在创建...")
            os.makedirs(outpath)
            print("创建成功！")
        for kk in k:
            out = outpath + sub_name + kk
            watershed_self(i + kk, out)
            print("第{}个图片创建完成".format(k.index(kk) + 1))


def test():
    watershed_self('./capacity/2.jpg')


if __name__ == '__main__':
    path = './capacity/'
    outpath = './watershed_out/'
    sub_name = 'out_'
    main()
    # test()
