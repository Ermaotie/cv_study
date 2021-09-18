import cv2
import os
import numpy as np


# 裁剪 目录下所有文件
def CropCenter4Dir(inpath, outpath, front='out_'):
    pathDir = os.listdir(inpath)  # 列出文件路径中的所有路径或文件
    if not os.path.exists(outpath):
        print("路径不存在，正在创建...")
        os.makedirs(outpath)
        print("创建成功！")
    for allDir in pathDir:
        child = os.path.join(inpath, allDir)
        dest = os.path.join(outpath, allDir)
        if os.path.isfile(child):
            image = cv2.imread(child)
            sp = image.shape  # 获取图像形状：返回【行数值，列数值】列表
            sz1 = sp[0]  # 图像的高度（行 范围）
            sz2 = sp[1]  # 图像的宽度（列 范围）
            # sz3 = sp[2]                #像素值由【RGB】三原色组成

            # 你想对文件的操作
            edge = 30
            a = int(edge)  # x start
            b = int(sz1 - edge)  # x end
            c = int(edge)  # y start
            d = int(sz2 - 30)  # y end
            cropImg = image[a:b, c:d]  # 裁剪图像
            cv2.imwrite(dest, cropImg)  # 写入图像路径


# 拉普拉斯锐化 array
def shapen_Laplacian(in_img):
    I = in_img.copy()
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    L = cv2.filter2D(I, -1, kernel)
    a = 0.5
    O = cv2.addWeighted(I, 1, L, a, 0)
    O[O > 255] = 255
    O[O < 0] = 0
    return O


